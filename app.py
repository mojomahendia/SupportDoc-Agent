import os

import streamlit as st

# Inject Streamlit secrets into environment before any graph imports trigger load_dotenv().
# Locally, load_dotenv() reads from .env — on Streamlit Cloud, st.secrets supplies the values.
for _key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "USER_AGENT"]:
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from graph.graph import graph  # noqa: E402  (must come after env injection)

st.set_page_config(page_title="SupportDoc Agent", page_icon="🛡️", layout="centered")
st.title("SupportDoc Agent")
st.caption("Microsoft Intune troubleshooting assistant — powered by agentic RAG")

query = st.text_input(
    "Ask a question about Microsoft Intune",
    placeholder="e.g. Why does enrollment error 80180014 occur?",
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Searching documentation..."):
        result = graph.invoke({"query": query.strip(), "retrieval_count": 0})
        st.session_state["last_result"] = result

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.markdown(r["generation"])
    attempts = r.get("retrieval_count", 0)
    st.caption(f"Retrieval attempts: {attempts}")
