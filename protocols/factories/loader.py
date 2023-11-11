# protocols/factories/loader.py

from typing import Protocol

from langchain.document_loaders.base import BaseLoader


class ILoaderFactory(Protocol):
    def create_loader(self) -> BaseLoader:
        ...
