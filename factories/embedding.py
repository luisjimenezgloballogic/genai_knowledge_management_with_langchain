# factories/embedding.py

from typing import Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings

from protocols.config import IConfigLoader


class OpenAIEmbeddingsFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config: Dict = config_loader.load()
        self.embedding_model = self.config["embedding"]["model"]["openai"]

    def create_embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(model=self.embedding_model)
