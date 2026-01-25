"""LangChain-based vector store initialization."""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.settings import settings


def get_vectorstore():
    """
    Get or create ChromaDB vector store using LangChain.

    Returns:
        Chroma vector store instance
    """
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(
        persist_directory=settings.rag_vector_db_path,
        embedding_function=embeddings,
    )

    return vectorstore

