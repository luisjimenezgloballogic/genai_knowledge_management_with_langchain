# protocols/factories/embedding.py

from typing import Protocol

from langchain.schema.embeddings import Embeddings


class IEmbeddingsFactory(Protocol):
    def create_embeddings(self) -> Embeddings:
        ...
