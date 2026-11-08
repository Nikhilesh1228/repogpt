"""
RepoGPT Codebase Indexer
Handles downloading, filtering, and parsing public GitHub repositories.
Works in serverless and cloud environments without needing the git CLI installed.
"""

import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests

# Directories to ignore
IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", ".env",
    "__pycache__", "dist", "build", ".next", ".nuxt", ".idea",
    ".vscode", "target", "vendor", "bin", "obj", ".mypy_cache",
    ".pytest_cache", "coverage", ".turbo", ".gradle", "Pods"
}

# File extensions to ignore (binaries, media, compiled assets, lockfiles)
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyo", ".wasm",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".pdf", ".docx", ".xlsx",
    ".lock", ".lockb", ".min.js", ".min.css", ".map",
    ".ttf", ".woff", ".woff2", ".eot", ".otf",
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    ".DS_Store", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"
}

# Mapping extensions to LangChain / Syntax Highlighter language names
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".mjs": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".sql": "sql",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".toml": "toml",
    ".xml": "xml",
    ".proto": "protobuf"
}

MAX_FILE_SIZE_BYTES = 120 * 1024  # 120 KB per file limit to filter out massive generated code


@dataclass
class CodeFile:
    relative_path: str
    file_name: str
    language: str
    content: str
    size_bytes: int
    line_count: int


@dataclass
class RepoStats:
    owner: str
    repo: str
    total_files: int
    total_lines: int
    languages: Dict[str, int]


def parse_github_url(url: str) -> Tuple[str, str]:
    """Extracts owner and repo from a standard GitHub URL."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    
    # Matches patterns like github.com/owner/repo
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Invalid GitHub repository URL: {url}. Expected format: https://github.com/owner/repo")
    
    owner, repo = match.group(1), match.group(2)
    return owner, repo


def download_github_repo(owner: str, repo: str, branch: Optional[str] = None) -> Tuple[bytes, str]:
    """
    Downloads repository archive from GitHub directly as a zipball.
    Tries specified branch or falls back to 'main' then 'master'.
    """
    branches_to_try = [branch] if branch else ["main", "master"]
    headers = {"User-Agent": "RepoGPT-Code-Assistant"}
    
    for b in branches_to_try:
        archive_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{b}.zip"
        try:
            response = requests.get(archive_url, headers=headers, stream=True, timeout=25)
            if response.status_code == 200:
                return response.content, b
        except requests.RequestException:
            continue
            
    # Fallback to GitHub zipball API endpoint
    api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    try:
        response = requests.get(api_url, headers=headers, stream=True, timeout=25)
        if response.status_code == 200:
            return response.content, "default"
    except requests.RequestException:
        pass
        
    raise RuntimeError(
        f"Could not download repository '{owner}/{repo}'. Please check that the repository is public and accessible."
    )


def should_ignore_path(rel_path: str) -> bool:
    """Checks if a file or directory path should be skipped."""
    parts = os.path.normpath(rel_path).split(os.sep)
    for part in parts:
        if part in IGNORED_DIRS or part.startswith(".git"):
            return True
            
    filename = parts[-1].lower()
    if filename in IGNORED_EXTENSIONS:
        return True
        
    _, ext = os.path.splitext(filename)
    if ext in IGNORED_EXTENSIONS:
        return True
        
    return False


def extract_code_files(zip_bytes: bytes) -> Tuple[List[CodeFile], RepoStats, str, str]:
    """
    Unpacks zip bytes in memory and parses relevant code files.
    Returns list of CodeFile, statistics, and repo info.
    """
    code_files: List[CodeFile] = []
    languages_count: Dict[str, int] = {}
    total_lines = 0
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        namelist = z.namelist()
        if not namelist:
            raise ValueError("The downloaded repository zip archive is empty.")
            
        # First entry is usually the root folder name e.g. "owner-repo-commit_sha/"
        root_dir = namelist[0].split("/")[0]
        
        for name in namelist:
            # Skip directories
            if name.endswith("/"):
                continue
                
            # Strip the root wrapper folder
            rel_path = name[len(root_dir) + 1:] if name.startswith(root_dir + "/") else name
            
            if should_ignore_path(rel_path):
                continue
                
            info = z.getinfo(name)
            # Skip files larger than limit
            if info.file_size > MAX_FILE_SIZE_BYTES or info.file_size == 0:
                continue
                
            _, ext = os.path.splitext(rel_path)
            ext_lower = ext.lower()
            
            # Identify language
            file_name = os.path.basename(rel_path)
            if file_name.lower() in ("dockerfile", "containerfile"):
                lang = "dockerfile"
            elif ext_lower in EXTENSION_LANGUAGE_MAP:
                lang = EXTENSION_LANGUAGE_MAP[ext_lower]
            else:
                # Only include unknown extensions if they look like plain text code
                continue
                
            try:
                raw_bytes = z.read(name)
                # Attempt decoding as UTF-8
                content = raw_bytes.decode("utf-8")
            except (UnicodeDecodeError, Exception):
                # Binary or non-UTF8 encoded file
                continue
                
            lines = content.count("\n") + 1
            total_lines += lines
            languages_count[lang] = languages_count.get(lang, 0) + 1
            
            code_files.append(
                CodeFile(
                    relative_path=rel_path,
                    file_name=file_name,
                    language=lang,
                    content=content,
                    size_bytes=info.file_size,
                    line_count=lines
                )
            )
            
    stats = RepoStats(
        owner="",
        repo="",
        total_files=len(code_files),
        total_lines=total_lines,
        languages=languages_count
    )
    
    return code_files, stats, "", ""
