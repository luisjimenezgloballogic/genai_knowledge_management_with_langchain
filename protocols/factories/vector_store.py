# protocols/factories/vector_store.py

from typing import Protocol

from langchain.document_loaders.base import BaseLoader
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import TextSplitter


class IVectorStoreFactory(Protocol):
    def create_vector_store(
        self, embeddings: Embeddings, loader: BaseLoader, text_splitter: TextSplitter
    ) -> VectorStore:
        ...

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        ...
