from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from groq import Groq
from web_tools import simple_search
from rag_store import RAGStore
import os

# ---- State Definition ----
class PipelineState(BaseModel):
    topic: str
    research_notes: List[str] = Field(default_factory=list)
    indexed: bool = False
    context_snippets: List[str] = Field(default_factory=list)
    final_report: Optional[str] = None
    error: Optional[str] = None

# ---- LLM Helper ----
_groq = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

def llm_chat(system: str, user: str, temperature: float = 0.0) -> str:
    resp = _groq.chat.completions.create(
        model=_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
    )
    return resp.choices[0].message.content  # ✅ FIXED


# ---- Nodes ----
def research_node(state: PipelineState) -> PipelineState:
    try:
        # E-commerce-focused research
        search_hits = simple_search(f"{state.topic} ecommerce competitor analysis", max_results=6)
        notes = []
        for hit in search_hits:
            summary = llm_chat(
                "Summarize this for e-commerce competitor intelligence (pricing, logistics, CX, market share).",
                hit
            )
            notes.append(summary)
        state.research_notes = notes
    except Exception as e:
        state.error = f"research_node failed: {e}"
    return state

def index_node(state: PipelineState, rag: RAGStore | None = None) -> PipelineState:
    if not rag:
        state.error = "RAG store not available"
        return state
    try:
        if not state.research_notes:
            state.indexed = False
            return state
        rag.add_texts(state.research_notes, metadata={"topic": state.topic})
        state.indexed = True
    except Exception as e:
        state.indexed = False
        state.error = f"index_node failed: {e}"
    return state

def draft_node(state: PipelineState, rag: RAGStore | None = None) -> PipelineState:
    try:
        ctx = rag.retrieve(state.topic, k=8) if rag else []
        state.context_snippets = ctx
        context_blob = "\n\n".join(ctx) if ctx else "NO_CONTEXT"
        prompt = f"""
Topic: {state.topic}

You are an expert analyst for the ecommerce industry.

Write a competitor analysis report including:
- Executive Summary
- Key Ecommerce Trends
- Major Competitors (Walmart, Alibaba, Flipkart, Target, etc.)
- Comparative Matrix (pricing, logistics, delivery time, app UX, product range)
- Strengths & Weaknesses of each competitor
- Opportunities & Risks for {state.topic}
- Strategic Recommendations (delivery, pricing, marketplace model, AI)

Use ONLY the context below. If context is weak, say so and keep claims conservative.

Context:
{context_blob}
"""
        report = llm_chat(
            "You produce accurate, concise reports for executives using provided context only.",
            prompt,
            temperature=0.1
        )
        state.final_report = report
    except Exception as e:
        state.error = f"draft_node failed: {e}"
    return state

# ---- Build the Graph ----
def build_graph(rag: RAGStore):
    graph = StateGraph(PipelineState)

    # inject RAG instance where needed
    def _index_node(state: PipelineState): return index_node(state, rag=rag)
    def _draft_node(state: PipelineState): return draft_node(state, rag=rag)

    graph.add_node("research", research_node)
    graph.add_node("index", _index_node)
    graph.add_node("draft", _draft_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "index")
    graph.add_edge("index", "draft")
    graph.add_edge("draft", END)
    return graph.compile()
