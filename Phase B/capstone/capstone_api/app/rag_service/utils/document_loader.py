"""Document loading utilities using LangChain."""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.settings import settings


def load_pdf(path: str | Path):
    """
    Load PDF using LangChain's PyPDFLoader.
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of LangChain Document objects
    """
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size: int | None = None, chunk_overlap: int | None = None):
    """
    Split documents into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Size of each chunk (defaults to settings value)
        chunk_overlap: Overlap between chunks (defaults to settings value)
        
    Returns:
        List of chunked Document objects
    """
    chunk_size = chunk_size or settings.rag_chunk_size
    chunk_overlap = chunk_overlap or settings.rag_chunk_overlap
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a helpful AI assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

def build_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

