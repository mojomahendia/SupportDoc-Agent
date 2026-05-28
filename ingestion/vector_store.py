
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()  # OPENAI_API_KEY picked up automatically from .env

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"   # persist — otherwise lost on restart
    )
    return vectorstore