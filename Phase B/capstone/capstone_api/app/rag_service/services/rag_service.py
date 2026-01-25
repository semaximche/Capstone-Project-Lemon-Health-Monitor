"""LangChain-based RAG service."""
from app.rag_service.llm.llm_client import get_llm_client
from uuid import UUID
from app.rag_service.db.vectorstore import get_vectorstore
from app.rag_service.utils.document_loader import build_prompt,build_context

class RAGService:

    def query(self,query: str) -> str:
        """
        Process a RAG query for a user.

        Args:
            query: User query string

        Returns:
            Generated response from LLM
        """

        try:
            # 1. Load vector DB
            vector_db = get_vectorstore()
            if vector_db is None:
                raise RuntimeError("Vector store is not initialized")

            # 2. Retrieve relevant docs
            retriever = vector_db.as_retriever()
            relevant_docs = retriever.invoke(query)

            if not relevant_docs:
                return "I couldn't find relevant information in the knowledge base."

            # 3. Build context & prompt
            context = build_context(relevant_docs)
            prompt = build_prompt(context, query)

            # 4. Generate answer
            client = get_llm_client()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            if not response:
                raise RuntimeError("LLM returned an empty response")

            return response.text

        except Exception as e:
            raise RuntimeError("Failing to retrieve data from rag , error: {}".format(e))


rag_service = RAGService()
