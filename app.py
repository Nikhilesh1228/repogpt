"""
RepoGPT – Chat With Any GitHub Repository
A Production-Ready Code-RAG Web Application.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from indexer import (
    parse_github_url,
    download_github_repo,
    extract_code_files
)
from rag_engine import (
    chunk_code_files,
    build_vector_store,
    query_codebase
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RepoGPT – Codebase Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished portfolio look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .citation-card {
        border-left: 3px solid #3B82F6;
        padding-left: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "repo_stats" not in st.session_state:
    st.session_state.repo_stats = None
if "active_repo" not in st.session_state:
    st.session_state.active_repo = None
if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}


# Helper to safely retrieve secrets across Local (.env) and Cloud (st.secrets)
def get_safe_secret(key: str, default: str = "") -> str:
    # 1. Check os.environ / .env first
    val = os.getenv(key)
    if val:
        return val
    # 2. Check st.secrets safely without throwing StreamlitSecretNotFoundError locally
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# --- SIDEBAR: Configuration & Ingestion ---
with st.sidebar:
    st.markdown("### ⚡ RepoGPT Settings")
    
    # 1. AI Provider Selection
    provider = st.selectbox(
        "AI Provider",
        options=["Gemini", "OpenAI"],
        index=0,
        help="Google Gemini offers generous free tier credits and 1M+ context window."
    )
    
    # API Key Handling (Check env / secrets first)
    secret_key_name = "GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY"
    env_key = get_safe_secret(secret_key_name)
        
    api_key = st.text_input(
        f"{provider} API Key",
        value=env_key,
        type="password",
        placeholder=f"Enter {provider} key here...",
        help="Your key is only kept in session memory and never stored."
    )
    
    st.markdown("---")
    st.markdown("### 📦 Repository Ingestion")
    
    # Quick pick examples
    sample_repo = st.selectbox(
        "Try an Example Repo or Enter Below",
        options=[
            "Custom URL",
            "https://github.com/pallets/flask",
            "https://github.com/fastapi/fastapi",
            "https://github.com/expressjs/express"
        ],
        index=0
    )
    
    default_url = "" if sample_repo == "Custom URL" else sample_repo
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        value=default_url,
        placeholder="https://github.com/owner/repository"
    )
    
    branch_input = st.text_input(
        "Branch (Optional)",
        value="",
        placeholder="main / master (default)"
    )
    
    index_button = st.button("🚀 Analyze & Index Codebase", use_container_width=True, type="primary")
    
    if index_button:
        if not api_key:
            st.error(f"Please provide a valid {provider} API key to proceed.")
        elif not repo_url_input:
            st.error("Please enter a valid GitHub repository URL.")
        else:
            try:
                owner, repo = parse_github_url(repo_url_input)
                progress_bar = st.progress(10, text="Fetching repository from GitHub...")
                
                # Step 1: Download
                zip_bytes, branch_used = download_github_repo(owner, repo, branch_input.strip() or None)
                progress_bar.progress(40, text=f"Parsing code structure & filtering assets (branch: {branch_used})...")
                
                # Step 2: Extract & Filter
                code_files, stats, _, _ = extract_code_files(zip_bytes)
                stats.owner = owner
                stats.repo = repo
                
                if not code_files:
                    st.error("No supported code files found in this repository.")
                else:
                    progress_bar.progress(70, text=f"Language-aware AST chunking ({len(code_files)} files)...")
                    chunks = chunk_code_files(code_files)
                    
                    progress_bar.progress(85, text=f"Generating embeddings & building FAISS index ({len(chunks)} chunks)...")
                    v_store = build_vector_store(chunks, provider, api_key)
                    
                    progress_bar.progress(100, text="Ready!")
                    
                    # Update State
                    st.session_state.vector_store = v_store
                    st.session_state.repo_stats = stats
                    st.session_state.active_repo = f"{owner}/{repo}"
                    st.session_state.chat_history = []
                    st.session_state.sources_history = {}
                    
                    st.success(f"Indexed **{owner}/{repo}** successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error indexing repository: {str(e)}")

    # Show Stats if active
    if st.session_state.repo_stats:
        st.markdown("---")
        st.markdown(f"**Indexed:** `{st.session_state.active_repo}`")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Code Files", st.session_state.repo_stats.total_files)
        with col2:
            st.metric("Total Lines", f"{st.session_state.repo_stats.total_lines:,}")
            
        st.markdown("**Languages Detected:**")
        langs_html = ""
        for lang, count in sorted(st.session_state.repo_stats.languages.items(), key=lambda x: -x[1])[:6]:
            langs_html += f'<span class="stat-badge">{lang}: {count}</span>'
        st.markdown(langs_html, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Indexed Codebase", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.repo_stats = None
            st.session_state.active_repo = None
            st.session_state.chat_history = []
            st.session_state.sources_history = {}
            st.rerun()


# --- MAIN CHAT & DASHBOARD AREA ---
st.markdown('<div class="main-header">⚡ RepoGPT – Codebase Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Autonomous architectural Q&A, function tracing, and bug inspection over any public GitHub repository.</div>',
    unsafe_allow_html=True
)

if not st.session_state.vector_store:
    # Landing / Getting Started Guide
    st.info("👈 **Get Started:** Enter your API key and paste any public GitHub repository in the left sidebar to begin.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 1. Paste Repo URL")
        st.write("Input any open-source GitHub repo. It automatically filters noise like `.git`, `node_modules`, and binary assets.")
    with col_b:
        st.markdown("#### 2. AST Code Chunking")
        st.write("Respects code syntax boundaries (classes, functions, and modules) rather than cutting mid-sentence.")
    with col_c:
        st.markdown("#### 3. Ask Anything")
        st.write("Ask about data flows, error handling, entry points, or generate unit tests with verifiable file citations.")
        
    st.markdown("---")
    st.markdown("#### 💡 Example Questions You Can Ask RepoGPT:")
    st.markdown("""
    - *"Explain the high-level architecture and how requests flow through this codebase."*
    - *"Where is authentication and token validation implemented?"*
    - *"List the primary database models and their relationships."*
    - *"Write pytest unit tests for the main utility functions."*
    """)

else:
    # Active Repo Header
    st.success(f"🟢 Active Codebase: **https://github.com/{st.session_state.active_repo}**")
    
    # Suggested Prompt Chips
    st.markdown("**Suggested Quick Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    preset_prompt = None
    with q_col1:
        if st.button("🏛️ Explain Architecture", use_container_width=True):
            preset_prompt = "Explain the high-level architecture, directory layout, and main flow of this project."
    with q_col2:
        if st.button("🚀 Find Main Entry Point", use_container_width=True):
            preset_prompt = "Where is the main entry point or server bootstrap of this project? How does execution start?"
    with q_col3:
        if st.button("⚠️ Error Handling & Config", use_container_width=True):
            preset_prompt = "How is environment configuration and global error handling implemented?"

    # Render Chat History
    for idx, (role, text) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.markdown(text)
            # Display citations if available for this assistant turn
            if role == "assistant" and idx in st.session_state.sources_history:
                citations = st.session_state.sources_history[idx]
                if citations:
                    with st.expander(f"📂 Referenced Code Sources ({len(citations)} files)"):
                        for c in citations:
                            st.markdown(f"**File:** `{c['source']}` *(Language: {c['language']})*")
                            st.code(c["code"], language=c["language"] if c["language"] in ["python", "js", "ts", "go", "cpp", "java", "html", "css"] else "text")

    # Chat Input Handling
    user_query = st.chat_input("Ask a question about this repository's code...") or preset_prompt

    if user_query:
        if not api_key:
            st.error(f"Please provide your {provider} API key in the sidebar.")
        else:
            # 1. Display User Message
            st.session_state.chat_history.append(("user", user_query))
            with st.chat_message("user"):
                st.markdown(user_query)
                
            # 2. Process with Code RAG
            with st.chat_message("assistant"):
                with st.spinner("Analyzing codebase and tracing code paths..."):
                    try:
                        result = query_codebase(
                            question=user_query,
                            vector_store=st.session_state.vector_store,
                            provider=provider,
                            api_key=api_key,
                            chat_history=st.session_state.chat_history[:-1]
                        )
                        
                        answer = result["answer"]
                        citations = result["citations"]
                        
                        st.markdown(answer)
                        
                        # Store in history
                        assistant_idx = len(st.session_state.chat_history)
                        st.session_state.chat_history.append(("assistant", answer))
                        st.session_state.sources_history[assistant_idx] = citations
                        
                        if citations:
                            with st.expander(f"📂 Referenced Code Sources ({len(citations)} files)"):
                                for c in citations:
                                    st.markdown(f"**File:** `{c['source']}` *(Language: {c['language']})*")
                                    st.code(c["code"], language=c["language"] if c["language"] in ["python", "js", "ts", "go", "cpp", "java", "html", "css"] else "text")
                                    
                    except Exception as e:
                        err_msg = f"Error generating answer: {str(e)}"
                        st.error(err_msg)
                        st.session_state.chat_history.append(("assistant", err_msg))
