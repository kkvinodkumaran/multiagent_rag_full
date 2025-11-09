import os
from dotenv import load_dotenv

# ---------------------------------------------------
# ✅ Load .env FIRST — before anything else
# ---------------------------------------------------
# Try to load from multiple possible locations
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),  # Parent directory
    os.path.join(os.path.dirname(__file__), ".env"),  # Current directory
    ".env"  # Relative to working directory
]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("Warning: No .env file found, using environment variables from system")
    # Load from system environment (Docker will provide these)
    load_dotenv()

print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
print("EMBED_MODEL:", os.getenv("EMBED_MODEL"))
print("CHROMA_DIR:", os.getenv("CHROMA_DIR"))

# ---------------------------------------------------
# ✅ AFTER env is loaded — now import modules
# ---------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
from rag_store import RAGStore
from graph import build_graph, PipelineState

# ---------------------------------------------------
# ✅ Use env vars here
# ---------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Multi-Agent RAG (Groq + LangGraph)")

# init RAG store and graph once
rag = RAGStore(persist_dir=CHROMA_DIR, embed_model_name=EMBED_MODEL)
workflow = build_graph(rag)

class AnalyzeRequest(BaseModel):
    topic: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    init = PipelineState(topic=req.topic)
    result = workflow.invoke(init.dict())
    return result
