# factories/generator.py

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStoreRetriever

from models.generator import ConversationalGenerator, QAGenerator
from protocols.config import IConfigLoader
from protocols.models.generator import IGenerator


class ConversationalGeneratorFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config = config_loader.load()
        self.verbose = self.config["chain"]["conversational"]["verbose"]

    def create_generator(
        self, retriever: VectorStoreRetriever, llm: BaseLanguageModel
    ) -> IGenerator:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, verbose=self.verbose
        )
        return ConversationalGenerator(chain)


class QAGeneratorFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config = config_loader.load()
        self.chain_type = self.config["chain"]["qa"]["chain_type"]

    def create_generator(
        self, retriever: VectorStoreRetriever, llm: BaseLanguageModel
    ) -> IGenerator:
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type=self.chain_type, retriever=retriever
        )
        return QAGenerator(chain)
