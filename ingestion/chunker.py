from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ingestion.loader import load_documents

docs = load_documents()

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []

    for doc_index, doc in enumerate(docs):
        chunks = splitter.split_documents([doc])

        for chunk_index, chunk in enumerate(chunks):
            # Extend metadata — original metadata is already there
            chunk.metadata["doc_index"]    = doc_index          # which document
            chunk.metadata["chunk_index"]  = chunk_index        # position within doc
            chunk.metadata["total_chunks"] = len(chunks)        # how many chunks this doc produced
            chunk.metadata["is_first"]     = chunk_index == 0
            chunk.metadata["is_last"]      = chunk_index == len(chunks) - 1

            all_chunks.append(chunk)

    return all_chunks