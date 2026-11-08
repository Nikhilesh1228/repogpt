"""
RepoGPT Code RAG Engine
Implements Language-Aware Code Chunking, In-Memory FAISS Vector Indexing,
and Multi-Model (Gemini & OpenAI) Code Retrieval with Syntax Citations.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from indexer import CodeFile

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Map language string to LangChain Language enum
LANG_ENUM_MAP = {
    "python": Language.PYTHON,
    "js": Language.JS,
    "ts": Language.TS,
    "java": Language.JAVA,
    "c": Language.CPP,
    "cpp": Language.CPP,
    "go": Language.GO,
    "rust": Language.RUST,
    "ruby": Language.RUBY,
    "php": Language.PHP,
    "csharp": Language.CSHARP,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "kotlin": Language.KOTLIN,
    "markdown": Language.MARKDOWN,
    "html": Language.HTML
}

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def chunk_code_files(files: List[CodeFile]) -> List[Document]:
    """
    Chunks code files respecting language syntax boundaries (functions, classes, blocks).
    """
    all_chunks: List[Document] = []
    
    # Cache splitters by language
    splitters: Dict[str, RecursiveCharacterTextSplitter] = {}
    
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for file in files:
        lang = file.language
        if lang in LANG_ENUM_MAP:
            if lang not in splitters:
                splitters[lang] = RecursiveCharacterTextSplitter.from_language(
                    language=LANG_ENUM_MAP[lang],
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP
                )
            splitter = splitters[lang]
        else:
            splitter = default_splitter
            
        file_chunks = splitter.split_text(file.content)
        
        for i, chunk_text in enumerate(file_chunks):
            # Prepend file path header to chunk for enhanced embedding context
            enriched_content = f"// File: {file.relative_path} (Language: {file.language})\n{chunk_text}"
            doc = Document(
                page_content=enriched_content,
                metadata={
                    "source": file.relative_path,
                    "file_name": file.file_name,
                    "language": file.language,
                    "chunk_id": i,
                    "raw_code": chunk_text
                }
            )
            all_chunks.append(doc)
            
    return all_chunks


def get_embeddings_model(provider: str, api_key: str):
    """Factory for embedding models (Gemini or OpenAI)."""
    if provider.lower() == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
    elif provider.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_llm(provider: str, api_key: str, model_name: Optional[str] = None):
    """Factory for LLM chat models."""
    if provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=api_key,
            temperature=0.2
        )
    elif provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        target_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=target_model,
            openai_api_key=api_key,
            temperature=0.2
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def build_vector_store(chunks: List[Document], provider: str, api_key: str) -> FAISS:
    """Builds an in-memory FAISS vector index from code chunks."""
    embeddings = get_embeddings_model(provider, api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


SYSTEM_PROMPT = """You are RepoGPT, an elite software architect and senior codebase intelligence assistant.
Your mission is to help developers, hiring managers, and engineers understand, explore, and debug the given codebase.

Context provided below contains relevant code snippets extracted from the repository:
---------------------
{context}
---------------------

Instructions:
1. Ground your answers strictly in the provided code context. Mention specific file paths (e.g. `src/auth.py`), classes, and functions whenever relevant.
2. If the user asks for architecture or data flow, provide clear step-by-step logic tracing.
3. If writing code suggestions, follow the patterns and conventions existing in this repository.
4. If the provided context does not contain the answer, explicitly state: "The indexed files do not contain enough information to answer this question" rather than guessing.
5. Format code blocks with appropriate language syntax highlighting.
"""


def query_codebase(
    question: str,
    vector_store: FAISS,
    provider: str,
    api_key: str,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Retrieves top relevant code chunks and generates an architectural answer with citations.
    """
    # 1. Retrieve top-k chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(question)
    
    # 2. Format context string
    context_parts = []
    citations = []
    
    for doc in relevant_docs:
        source_path = doc.metadata.get("source", "Unknown file")
        lang = doc.metadata.get("language", "text")
        raw_code = doc.metadata.get("raw_code", doc.page_content)
        
        context_parts.append(doc.page_content)
        citations.append({
            "source": source_path,
            "language": lang,
            "code": raw_code
        })
        
    context_str = "\n\n".join(context_parts)
    
    # 3. Format prompt with history
    messages = [
        ("system", SYSTEM_PROMPT.format(context=context_str))
    ]
    
    if chat_history:
        for user_msg, ai_msg in chat_history[-4:]:  # Keep recent context
            messages.append(("human", user_msg))
            messages.append(("ai", ai_msg))
            
    messages.append(("human", question))
    
    # 4. Generate answer
    llm = get_llm(provider, api_key)
    response = llm.invoke(messages)
    
    return {
        "answer": response.content,
        "citations": citations
    }
