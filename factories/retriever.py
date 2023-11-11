# factories/retriever.py

from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever

from protocols.config import IConfigLoader


class NumDocsRetrieverFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config = config_loader.load()
        self.num_docs = self.config["retriever"]["search_kwargs"]["k"]

    def create_retriever(self, vector_db: VectorStore) -> VectorStoreRetriever:
        return vector_db.as_retriever(
            search_kwargs={
                "k": self.num_docs,
            }
        )
