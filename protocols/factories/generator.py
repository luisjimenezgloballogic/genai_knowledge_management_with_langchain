# protocols/factories/generator.py

from typing import Protocol

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStoreRetriever

from protocols.models.generator import IGenerator


class IGeneratorFactory(Protocol):
    def create_generator(
        self, retriever: VectorStoreRetriever, llm: BaseLanguageModel
    ) -> IGenerator:
        ...
