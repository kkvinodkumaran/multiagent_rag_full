import os
import httpx
import streamlit as st

BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
BASE = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

st.set_page_config(page_title="Multi-Agent RAG", page_icon="🧠", layout="centered")
st.title("🧠 Multi-Agent RAG (Groq + LangGraph) — E-commerce Focus")

topic = st.text_input("Enter a topic (e.g., 'Amazon competitor analysis')", "Amazon competitor analysis")

if st.button("Run Analysis"):
    with st.spinner("Running pipeline..."):
        try:
            r = httpx.post(f"{BASE}/analyze", json={"topic": topic}, timeout=300.0)
            r.raise_for_status()
            data = r.json()
            st.subheader("Executive Report")
            st.write(data.get("final_report") or "No report")

            with st.expander("Context used (snippets)"):
                for i, c in enumerate(data.get("context_snippets", []), 1):
                    st.markdown(f"**{i}.** {c}")

            with st.expander("Research notes"):
                for i, n in enumerate(data.get("research_notes", []), 1):
                    st.markdown(f"**{i}.** {n}")

            if data.get("error"):
                st.error(data["error"])
        except Exception as e:
            st.error(f"Error: {e}")
