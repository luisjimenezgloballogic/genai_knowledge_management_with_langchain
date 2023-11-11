# protocols/factories/retriever.py

from typing import Protocol

from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever


class IRetrieverFactory(Protocol):
    def create_retriever(self, vector_db: VectorStore) -> VectorStoreRetriever:
        ...
