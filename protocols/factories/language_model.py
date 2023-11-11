# protocols/factories/language_model.py

from typing import Protocol

from langchain.schema.language_model import BaseLanguageModel


class ILanguageModelFactory(Protocol):
    def create_language_model(self) -> BaseLanguageModel:
        ...
