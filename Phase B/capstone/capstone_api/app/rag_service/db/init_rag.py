from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from app.settings import settings
from google import genai

vector_db_path = Path(settings.rag_vector_db_path)

if not vector_db_path.exists():
    print("not exists")

loader = DirectoryLoader(
    "C:\\Users\\david_k\\Desktop\\magshimim\\temp",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings()

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=settings.rag_vector_db_path
)

vector_db.persist()
print("Vector database initialized from PDFs and stored.")

