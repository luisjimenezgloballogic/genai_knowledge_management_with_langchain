# LangChain Classes:
# BaseLoader class in LangChain is a base class for loading documents in LangChain. It provides common functions for downloading and processing text.
# RecursiveCharacterTextSplitter class in LangChain recursively splits text into smaller chunks based on a list of separator characters until a desired chunk size is reached.
# TextSplitter class in LangChain is an abstract class that defines a common interface for splitting text into smaller chunks or documents.
# OpenAIEmbeddings class in Langchain allows generating text embeddings using OpenAI's models. Requires an OpenAI API key.
# Embeddings class in LangChain generates vector representations of text using embedding models to enable semantic search and comparisons.
# Document class in LangChain represents a document containing raw text and metadata. It is used in docstores to store and retrieve documents.
# VectorStore class in LangChain stores embedding vectors for fast semantic search. It allows indexing and retrieving similar documents.
# Chroma class in LangChain allows storing vectors using Anthropic's Chroma library. It enables fast vector lookups.
# VectorStoreRetriever is a retriever that uses a VectorStore to search for documents similar to a query. It wraps a VectorStore to conform to the retriever interface. It uses methods like similarity and MMR to query texts.
# BaseLanguageModel is a common interface for language models. It defines predict methods for text models and predict_messages for conversational models. Allows swapping different models while maintaining the same API.
# ChatOpenAI is a conversational model from OpenAI. It wraps the OpenAI API to predict responses in a conversation. It provides a standardized interface for OpenAI conversational models.
# Chain class in LangChain allows chaining multiple processing steps to build complex pipelines. It provides a standardized interface for joining simple components and creating processing chains with conditional and customizable flows. 
# ConversationalRetrievalChain class in LangChain enables conversational document retrieval. It condenses the chat history and current question into a query to retrieve relevant documents. It then passes those documents to a language model to generate a conversational response.
# RetrievalQA class in LangChain enables question answering based on document retrieval. It uses a retriever to search for documents relevant to the user's question. It then passes those documents to a language model to generate a conversational response.

# ====== main module =======

# main.py

from typing import Dict, List

from langchain.document_loaders.base import BaseLoader
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain.text_splitter import TextSplitter

from config import YamlConfigLoader
from factories.data_store import JSONLDataStoreFactory
from factories.embedding import OpenAIEmbeddingsFactory
from factories.extractor import GitHubExtractorFactory
from factories.generator import ConversationalGeneratorFactory, QAGeneratorFactory
from factories.gui import CommandLineChatGUIFactory
from factories.language_model import ChatOpenAIFactory
from factories.loader import JSONLLoaderFactory
from factories.preprocessor import TextCleanerFactory
from factories.retriever import NumDocsRetrieverFactory
from factories.text_splitter import RecursiveCharacterTextSplitterFactory
from factories.vector_store import ChromaFactory
from protocols.factories.data_store import IDataStoreFactory
from protocols.factories.embedding import IEmbeddingsFactory
from protocols.factories.extractor import IExtractorFactory
from protocols.factories.generator import IGeneratorFactory
from protocols.factories.gui import IGUIFactory
from protocols.factories.language_model import ILanguageModelFactory
from protocols.factories.loader import ILoaderFactory
from protocols.factories.preprocessor import IDataPreprocessorFactory
from protocols.factories.retriever import IRetrieverFactory
from protocols.factories.text_splitter import ITextSplitterFactory
from protocols.factories.vector_store import IVectorStoreFactory
from protocols.models.data_store import IDataStore
from protocols.models.extractor import IExtractor
from protocols.models.generator import IGenerator
from protocols.models.gui import IGUI
from protocols.models.preprocessor import IDataPreprocessor


class Application:
    def __init__(
        self,
        extractor_factory: IExtractorFactory,
        data_preprocessor_factory: IDataPreprocessorFactory,
        data_store_factory: IDataStoreFactory,
        loader_factory: ILoaderFactory,
        text_splitter_factory: ITextSplitterFactory,
        embeddings_factory: IEmbeddingsFactory,
        vector_store_factory: IVectorStoreFactory,
        retriever_factory: IRetrieverFactory,
        language_model_factory: ILanguageModelFactory,
        generator_factory: IGeneratorFactory,
        gui_factory: IGUIFactory,
    ) -> None:
        self.extractor_factory = extractor_factory
        self.data_preprocessor_factory = data_preprocessor_factory
        self.data_store_factory = data_store_factory
        self.loader_factory = loader_factory
        self.text_splitter_factory = text_splitter_factory
        self.embeddings_factory = embeddings_factory
        self.vector_store_factory = vector_store_factory
        self.retriever_factory = retriever_factory
        self.language_model_factory = language_model_factory
        self.generator_factory = generator_factory
        self.gui_factory = gui_factory

    def store_data(self):
        data_store: IDataStore = self.data_store_factory.create_data_store()
        data_preprocessor: IDataPreprocessor = (
            self.data_preprocessor_factory.create_data_preprocessor()
        )
        source_list: List[Dict] = self.extractor_factory.get_source_list()
        for source in source_list:
            extractor: IExtractor = self.extractor_factory.create_extractor(source)
            extractor.extract(data_preprocessor, data_store)

    def store_vector(self):
        loader: BaseLoader = self.loader_factory.create_loader()
        text_splitter: TextSplitter = self.text_splitter_factory.create_text_splitter()
        embeddings: Embeddings = self.embeddings_factory.create_embeddings()
        vector_store: VectorStore = self.vector_store_factory.create_vector_store(
            embeddings, loader, text_splitter
        )
        return vector_store

    def chat(self):
        embeddings: Embeddings = self.embeddings_factory.create_embeddings()
        vector_store: VectorStore = self.vector_store_factory.get_vector_store(
            embeddings
        )
        retriever: VectorStoreRetriever = self.retriever_factory.create_retriever(
            vector_store
        )
        llm: BaseLanguageModel = self.language_model_factory.create_language_model()
        gui: IGUI = self.gui_factory.create_gui(retriever, llm)
        generator: IGenerator = self.generator_factory.create_generator(retriever, llm)
        gui.run(generator)


if __name__ == "__main__":
    config_loader = YamlConfigLoader()
    app = Application(
        extractor_factory=GitHubExtractorFactory(config_loader),
        data_store_factory=JSONLDataStoreFactory(config_loader),
        data_preprocessor_factory=TextCleanerFactory(),
        loader_factory=JSONLLoaderFactory(config_loader),
        text_splitter_factory=RecursiveCharacterTextSplitterFactory(config_loader),
        embeddings_factory=OpenAIEmbeddingsFactory(config_loader),
        vector_store_factory=ChromaFactory(config_loader),
        retriever_factory=NumDocsRetrieverFactory(config_loader),
        language_model_factory=ChatOpenAIFactory(config_loader),
        generator_factory=ConversationalGeneratorFactory(config_loader), # uncomment with Memory Chat
        # generator_factory=QAGeneratorFactory(config_loader), # uncomment with Stateless Chat
        gui_factory=CommandLineChatGUIFactory(config_loader),
    )
    # app.store_data() # uncomment to extract the data source
    # app.store_vector() # uncomment to create the vector database
    app.chat()
