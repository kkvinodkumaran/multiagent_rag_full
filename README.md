# Multi-Agent RAG System  
### (Groq + LangGraph + FastAPI + Streamlit + ChromaDB — powered by `uv` & Docker)

A fully local, production-style **Multi-Agent RAG workflow**, designed for backend engineers, ML enthusiasts, and system design learners.

This project demonstrates how to build a **real-world multi-agent system** with retrieval, summarization, and report generation — using only one external API (**Groq**, free key).

---

# ✅ 1. What Problem Are We Solving?

### ✅ **Use Case: E-commerce Competitor Analysis**
Companies often need to compare themselves with major competitors (Amazon, Flipkart, Walmart, Alibaba). Doing this manually requires:

- Searching online sources  
- Collecting competitor information  
- Summarizing insights  
- Combining everything into a structured report  

**This is slow, manual, and error-prone**.

### ✅ **Our Multi-Agent RAG System Automates This**
Given a topic like:

> **“Amazon competitor analysis”**

The system will:

1️⃣ Perform targeted web research  
2️⃣ Summarize each result  
3️⃣ Store the research in a vector database  
4️⃣ Retrieve context  
5️⃣ Generate a final executive-level analytical report  

All orchestrated via a **LangGraph state machine**.

---

# ✅ 2. Architecture Overview

```
User Input → Research Agent → Index Agent → Draft Agent → Final RAG Report
```

Each step is a **LangGraph node** with defined responsibilities.

```
┌───────────────┐
│   User Input   │
│  "Nike ecommerce" 
└───────┬─────────┘
        ▼
┌───────────────────────┐
│  Research Agent        │
│  - Web search          │
│  - LLM summaries       │
└───────┬────────────────┘
        ▼
┌───────────────────────┐
│  Index Agent           │
│  - Embed text          │
│  - Store in Chroma     │
└───────┬────────────────┘
        ▼
┌───────────────────────┐
│  Draft Agent           │
│  - Retrieve context    │
│  - Generate report     │
└───────┬────────────────┘
        ▼
┌───────────────────────┐
│ Final Competitor Report│
└────────────────────────┘
```

---

# ✅ 3. Tools Used in the System

## ✅ LLM Tools

| Tool | Purpose |
|------|---------|
| **Groq Llama 3** | Ultra-fast summarization & report generation |
| **llm_chat wrapper** | Safe wrapper around Groq chat.completions |

## ✅ Retrieval Tools

| Tool | Purpose |
|------|---------|
| **simple_search (Tavily)** | Web search to fetch snippets |
| **HuggingFaceEmbeddings** | Convert text → embeddings |
| **ChromaDB** | Local vector database for retrieval |

---

# ✅ 4. Agents & What Tools They Use

| Agent | Node Name | Responsibilities | Tools Used |
|--------|-----------|------------------|------------|
| **Research Agent** | `research` | Web search + summarization | `simple_search`, `llm_chat` |
| **Index Agent** | `index` | Embed text + store in vector DB | HF Embeddings, Chroma `add_texts()` |
| **Draft Agent** | `draft` | Retrieve context + generate final report | Chroma `retrieve()`, `llm_chat` |

---

# ✅ 5. LangGraph State Machine

### ✅ State Model

```python
class PipelineState(BaseModel):
    topic: str
    research_notes: List[str] = []
    indexed: bool = False
    context_snippets: List[str] = []
    final_report: Optional[str] = None
    error: Optional[str] = None
```

### ✅ Graph Nodes

| Node | Purpose |
|------|---------|
| `research` | Fetch data + summarize |
| `index` | Embed + store |
| `draft` | RAG + final report |

### ✅ Graph Structure

```
START → research → index → draft → END
```

---

# ✅ 6. Complete Tech Stack

- Python 3.11  
- uv  
- FastAPI  
- Streamlit  
- LangGraph  
- Groq LLM  
- ChromaDB  
- HuggingFace Sentence Transformers  

---

# ✅ 7. Running the System

## 📦 Prereqs

- Docker + Docker Compose  
- Groq API key  

---

## ⚙️ Setup Environment

You only need **one** `.env` file depending on how you run the system:

### Option 1: Docker Compose (Recommended)

Create `.env` in the **root directory**:
```bash
cp .env.example .env
```

Edit the root `.env` file with your API keys:
```bash
GROQ_API_KEY=gsk_xxx
TAVILY_API_KEY=tvly_xxx
MODEL_NAME=llama-3.1-8b-instant
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./data/chroma
BACKEND_HOST=backend
BACKEND_PORT=8000
```

### Option 2: Local Development

Create `.env` in the **backend/** directory:
```bash
cp .env.example backend/.env
```

Edit `backend/.env` with your API keys:
```bash
GROQ_API_KEY=gsk_xxx
TAVILY_API_KEY=tvly_xxx
MODEL_NAME=llama-3.1-8b-instant
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./data/chroma
```

**Note**: You don't need both files - choose the location based on your deployment method.

---

# Run With Docker (Recommended)

```
docker compose up --build
```

Open:

- Backend: http://localhost:8000/docs  
- UI: http://localhost:8501  

---

# 🛠 Local Dev (Optional)

### Backend
```
cd backend
uv sync
uv run uvicorn app:app --reload --port 8000
```

### UI
```
cd ui
uv sync
uv run streamlit run streamlit_app.py
```

---

# ✅ 8. Notes

- ChromaDB persists to `./data/chroma`  
- Replace Groq with Ollama for full offline mode  
- Extend system with more agents (Planner, Validator, Router)  

---

# ✅ 9. Summary

This project demonstrates a real, production-grade:

✅ Multi-Agent workflow  
✅ Retrieval-Augmented Generation  
✅ LangGraph orchestration  
✅ Chroma-based local vector retrieval  
✅ Groq Llama inference pipeline  
✅ Fully local deployment via Docker  

