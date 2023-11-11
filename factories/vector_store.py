# factories/vector_store.py

from langchain.document_loaders.base import BaseLoader
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma

from config import OpenAIAPIKeyLoader
from protocols.config import IConfigLoader


class ChromaFactory:
    def __init__(self, config_loader: IConfigLoader):
        config_loader = OpenAIAPIKeyLoader(config_loader)
        self.config = config_loader.load()
        self.path = self.config["embedding"]["store"]["chroma"]["path"]

    def create_vector_store(
        self, embeddings: Embeddings, loader: BaseLoader, text_splitter: TextSplitter
    ) -> VectorStore:
        documents = loader.load()
        splitted_documents = text_splitter.split_documents(documents)
        return Chroma.from_documents(
            documents=splitted_documents,
            embedding=embeddings,
            persist_directory=self.path,
        )

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        return Chroma(persist_directory=self.path, embedding_function=embeddings)
