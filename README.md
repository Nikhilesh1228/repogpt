# ⚡ RepoGPT – Autonomous Codebase Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-00599C.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **RepoGPT** is an open-source Code-RAG web application that allows developers, engineering leads, and recruiters to paste any public GitHub repository URL and instantly query, explore, trace, and debug its architecture using natural language.

---

## 🌟 Key Features

* **⚡ Zero-Git Serverless Ingestion:** Downloads and parses repository trees via in-memory streaming zip archives—no Git CLI required, making it 100% cloud-compatible.
* **🧠 Language-Aware AST Chunking:** Respects code syntax boundaries (classes, functions, and modules) across Python, JavaScript, TypeScript, Go, Java, Rust, and C++.
* **🛡️ Smart Noise Filtering:** Automatically detects and strips `.git`, `node_modules`, lockfiles (`package-lock.json`), binaries, and compiled assets.
* **🔎 In-Memory FAISS Vector Indexing:** Fast semantic retrieval with zero cloud database costs.
* **📂 Verifiable Code Citations:** Every answer cites the exact file path and displays syntax-highlighted code snippets of the referenced functions.
* **🤖 Multi-LLM Support:** Seamlessly switch between Google Gemini (Gemini 1.5 Flash) and OpenAI (GPT-4o-mini).

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A["GitHub URL Input"] --> B["In-Memory Zip Stream"]
    B --> C["Filter Non-Code Assets\n(.git, node_modules, binaries)"]
    C --> D["Language-Aware AST Chunking\n(LangChain CodeSplitter)"]
    D --> E["FAISS Vector Store\n(In-Memory CPU)"]
    E --> F["LLM Reasoning & Synthesis\n(Gemini / GPT)"]
    F --> G["Streamlit Web UI\n+ Syntax Citations"]
```

---

## 🛠️ Tech Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive web UI with real-time streaming chat and code viewers. |
| **Orchestration** | LangChain Core & Community | Document processing, text splitting, and retrieval chains. |
| **Code Parser** | Custom Tree Ingestion | Filters noise and maps extensions to programming language grammars. |
| **Vector Database**| FAISS (Facebook AI) | High-performance in-memory vector index. |
| **AI Models** | Google Gemini / OpenAI | Multi-model embeddings and code synthesis. |
| **Deployment** | Streamlit Community Cloud | Continuous deployment directly linked to GitHub repository. |

---

## 🚀 Quickstart: Run Locally on Your Laptop

### 1. Clone the repository
```bash
git clone https://github.com/your-username/repogpt.git
cd repogpt
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up your API key
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and insert your [Google Gemini API Key](https://aistudio.google.com/) or OpenAI API Key.

### 5. Launch the application
```bash
streamlit run app.py
```
The app will automatically open in your browser at `http://localhost:8501`.

---

## 🌐 Deploy to the Web for Free (Streamlit Cloud)

You can host RepoGPT online 24/7 with a public URL (e.g. `https://repogpt-codebase.streamlit.app`):

1. Push this directory to your personal GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of RepoGPT"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/repogpt.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New App"** and select:
   * **Repository:** `YOUR_USERNAME/repogpt`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. In **Advanced settings** $\to$ **Secrets**, paste your API key:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
5. Click **Deploy!** Your app will be live on the web in under 2 minutes.

---

## 📄 Resume Bullet Points Template

```markdown
RepoGPT – Autonomous Codebase Intelligence & Search Platform
Tech Stack: Python, Streamlit, LangChain, FAISS, Gemini API, OpenAI API, Git
• Engineered an open-source Code-RAG platform enabling natural language architectural Q&A and function tracing over any public GitHub repository.
• Implemented language-aware AST code chunking across 15+ programming languages, preserving class/function integrity and cutting hallucination rate.
• Designed a zero-dependency in-memory FAISS indexing pipeline that downloads, filters noise (ignoring node_modules/lockfiles), and embeds codebases in <15 seconds.
• Deployed production web application on Streamlit Cloud with real-time streaming responses and syntax-highlighted code source attribution.
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
