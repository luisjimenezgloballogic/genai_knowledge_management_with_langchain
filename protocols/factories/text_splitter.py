# protocols/factories/text_splitter.py

from typing import Protocol

from langchain.text_splitter import TextSplitter


class ITextSplitterFactory(Protocol):
    def create_text_splitter(self) -> TextSplitter:
        ...
