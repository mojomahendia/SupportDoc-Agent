from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.vector_store import build_vectorstore

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
