# factories/text_splitter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

from protocols.config import IConfigLoader


class RecursiveCharacterTextSplitterFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config = config_loader.load()["embedding"]["text_splitter"][
            "recursive_character"
        ]
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]

    def create_text_splitter(self) -> TextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            length_function=len,
            chunk_overlap=self.chunk_overlap,
        )
