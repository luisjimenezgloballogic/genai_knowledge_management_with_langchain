# protocols/factories/gui.py

from typing import Protocol

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStoreRetriever

from protocols.models.gui import IGUI


class IGUIFactory(Protocol):
    def create_gui(self, retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> IGUI:
        ...
