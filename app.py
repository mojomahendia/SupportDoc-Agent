import os
from pathlib import Path

import streamlit as st

try:
    for _key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2",
                 "LANGCHAIN_PROJECT", "USER_AGENT"]:
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from graph.graph import graph  # noqa: E402

st.set_page_config(
    page_title="SupportDoc Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

/* ── Layout ─────────────────────────────── */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ── Sidebar ────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #F0F4FF;
    border-right: 1px solid #DDEEFF;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #90A4AE;
    margin: 1.2rem 0 0.5rem 0;
}
/* Sidebar sample-question buttons — ghost style */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: white !important;
    color: #1565C0 !important;
    border: 1px solid #BBDEFB !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.45rem 0.8rem !important;
    margin-bottom: 0.2rem !important;
    transition: background 0.15s, border-color 0.15s !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: #E3F2FD !important;
    border-color: #1565C0 !important;
}

/* ── Main ask button ────────────────────── */
div[data-testid="column"] div[data-testid="stButton"] button,
.main-area div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1565C0 0%, #1E88E5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 0.65rem 2rem !important;
    transition: opacity 0.2s !important;
}
div[data-testid="column"] div[data-testid="stButton"] button:hover {
    opacity: 0.86 !important;
}

/* ── Text input ─────────────────────────── */
div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 2px solid #CFD8DC !important;
    padding: 0.8rem 1.1rem !important;
    font-size: 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #1565C0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.12) !important;
}

/* ── Chat message ───────────────────────── */
div[data-testid="stChatMessage"] {
    background: #F8FBFF !important;
    border: 1px solid #E3EEF9 !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.5rem !important;
}

/* ── Expander (sources) ─────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid #E3EEF9 !important;
    border-radius: 10px !important;
    background: #FAFCFF !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #1565C0;
}

/* ── Status caption ─────────────────────── */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: #90A4AE;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.4px;
}
.pill-blue   { background:#E3F2FD; color:#1565C0; }
.pill-green  { background:#E8F5E9; color:#2E7D32; }
.pill-amber  { background:#FFF8E1; color:#E65100; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=130)

    st.markdown("**SupportDoc Agent** answers Microsoft Intune troubleshooting questions using an agentic RAG pipeline — it retrieves documentation, grades relevance, rewrites queries if needed, and cites every claim.")

    st.markdown("### Try these")
    samples = [
        "Why does enrollment error 80180014 occur?",
        "Profile installation failed on iOS enrollment",
        "App protection policy not applying to user",
        "SCEP certificate profile deployment failing",
        "Configuration profile shows Conflict status",
        "Co-management auto-enrollment failing",
    ]
    for q in samples:
        if st.button(q, key=f"s_{q[:18]}", use_container_width=True):
            st.session_state["query_input"] = q
            st.rerun()

    st.markdown("### Stack")
    st.markdown(
        "LangGraph &nbsp;·&nbsp; LangChain  \n"
        "ChromaDB &nbsp;·&nbsp; GPT-4o-mini  \n"
        "RAGAs &nbsp;·&nbsp; LangSmith  \n"
        "Streamlit &nbsp;·&nbsp; Python 3.11",
        unsafe_allow_html=True,
    )

    st.markdown("### Links")
    st.markdown("[GitHub ↗](https://github.com/mojomahendia/SupportDoc-Agent)")

# ── Main ──────────────────────────────────────────────────────────────────────
_, main_col, _ = st.columns([1, 5, 1])

with main_col:
    # Hero
    h1, h2, h3 = st.columns([1, 2, 1])
    with h2:
        if logo_path.exists():
            st.image(str(logo_path), width=150)

    st.markdown(
        """
        <div style="text-align:center;margin-bottom:1.6rem;">
          <h1 style="font-size:2.4rem;font-weight:800;color:#0D47A1;
                     margin:0.3rem 0 0.25rem;letter-spacing:-0.5px;">
            SupportDoc Agent
          </h1>
          <p style="color:#607D8B;font-size:1.02rem;margin:0;">
            Microsoft Intune troubleshooting assistant &nbsp;·&nbsp; powered by agentic RAG
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Input — keyed so sidebar prefill can write to it
    query: str = st.text_input(
        "question",
        key="query_input",
        placeholder="e.g. Why does enrollment error 80180014 occur in Intune?",
        label_visibility="collapsed",
    )

    ask = st.button("Ask SupportDoc Agent", type="primary", use_container_width=True)

    if ask and query.strip():
        with st.spinner("Searching Intune documentation…"):
            result = graph.invoke({"query": query.strip(), "retrieval_count": 0})
            st.session_state["last_result"] = result

    # ── Answer ────────────────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        generation: str = r["generation"]
        attempts: int = r.get("retrieval_count", 0)
        route: str = r.get("route", "retrieve")

        _FALLBACK = "I wasn't able to find relevant information"
        is_fallback = generation.startswith(_FALLBACK)

        # Split answer from sources block
        if "[Sources]" in generation:
            answer_body, sources_raw = generation.split("[Sources]", 1)
        else:
            answer_body, sources_raw = generation, ""

        answer_body = answer_body.strip()

        # Parse sources → [(title, url)]
        sources: list[tuple[str, str]] = []
        for line in sources_raw.strip().splitlines():
            line = line.strip().lstrip("- ")
            if " — " in line:
                t, u = line.split(" — ", 1)
                sources.append((t.strip(), u.strip()))

        st.markdown("&nbsp;", unsafe_allow_html=True)

        # Chat-style answer — st.chat_message renders Markdown properly
        with st.chat_message("assistant", avatar="🛡️"):
            if is_fallback:
                st.warning(answer_body)
            else:
                st.markdown(answer_body)

            if sources:
                with st.expander(f"📚 Sources ({len(sources)})", expanded=True):
                    for title, url in sources:
                        st.markdown(f"[↗ {title}]({url})")

        # Status pill
        if route == "direct_answer":
            pill = '<span class="pill pill-green">✓ Direct answer</span>'
            detail = "No retrieval needed"
        elif is_fallback:
            pill = '<span class="pill pill-amber">⚠ Not in corpus</span>'
            detail = f"{attempts} retrieval attempt{'s' if attempts != 1 else ''} made"
        else:
            pill = '<span class="pill pill-blue">⬡ Retrieved</span>'
            detail = f"{attempts} retrieval attempt{'s' if attempts != 1 else ''}"

        st.markdown(
            f'<div class="status-row">{pill}'
            f'<span>{detail}</span></div>',
            unsafe_allow_html=True,
        )
