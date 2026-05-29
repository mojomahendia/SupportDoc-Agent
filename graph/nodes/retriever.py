from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma  # must match ingestion/vector_store.py
from langchain_openai import OpenAIEmbeddings

from graph.state import SupportDocState

load_dotenv()

# Absolute path — avoids cwd sensitivity when process starts outside project root
_CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

# Module-level singletons — constructed once on first import, reused across invocations
# Model must be identical to ingestion/vector_store.py: "text-embedding-3-small"
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(persist_directory=str(_CHROMA_DIR), embedding_function=_embeddings)


def retriever_node(state: SupportDocState) -> dict:
    search_query = state.get("rewritten_query") or state["query"]
    docs = _vectorstore.similarity_search(search_query, k=4)
    return {
        "documents": docs,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }
