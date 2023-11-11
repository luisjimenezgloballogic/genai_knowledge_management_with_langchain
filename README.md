# GenAI Knowledge Management with LangChain

## Summary

LangChain emerges as an innovative framework tailored for the development of smart search applications leveraging advanced language models such as those from OpenAI, Llama, and Cohere. Its modular and scalable structure is ideal for creating dynamic applications that integrate sophisticated language comprehension capabilities. LangChain's architecture allows for the customized assembly of processing chains, adaptable to the specific needs of an organization, with easy integration of document loaders, text transformers, and vector storage systems for efficient searches.

The project demonstrates the efficacy of LangChain through the implementation of the Retrieval-Augmented Generation (RAG) approach, particularly for question-answering tasks over a document corpus. The process involves loading, splitting, storing, retrieving, and generating information, structured in a pipeline that processes natural language queries with high efficiency.

In practice, the project showcases modularity across various workflow components such as data extraction, embedding generation, query response, and user interface creation. This modular design ensures maintainability and scalability, allowing for easy modification and extension without overhauling the application.

The results highlight the strengths and weaknesses of the `QAGenerator` and `ConversationalGenerator`. The `QAGenerator` provided accurate information but lacked conversational memory, leading to disconnected responses in follow-up queries. The `ConversationalGenerator`, however, maintained dialogue coherence, underscoring the importance of conversational memory in user interactions.

Lastly, the comparison with ChatGPT 3.5 underlines the advantage of having an up-to-date database, as LangChain-powered generators could offer current information, unlike ChatGPT 3.5 which is limited by its last knowledge update. This project illustrates LangChain’s role in developing AI communication systems with real-time data retrieval capabilities, offering a considerable advantage in the development of context-aware NLP applications.

## LangChain: A Versatile Framework

In today's era, marked by the imperative for rapid and accurate access to relevant information, LangChain emerges as an innovative framework for creating smart search applications powered by advanced language models such as OpenAI, Llama, Cohere, among others (LangChain, n.d.-a). LangChain enables the seamless integration of sophisticated language comprehension capabilities into tailored applications, enhancing the retrieval and contextualization of information.

For instance, an organization might utilize LangChain to develop an internal virtual assistant capable of processing natural language queries and retrieving specific answers from its corporate knowledge base. The language model leverages internal documentation to grasp the context and unique terminology of the entity, thus providing pertinent results. Developers can swiftly customize the system using LangChain’s flexible architecture to meet the ever-changing needs.

LangChain offers a modular and scalable framework for building dynamic applications driven by language models. Its architecture of interchangeable components allows developers to assemble customized processing chains tailored to their specific needs.

For example, LangChain's document loaders can be effortlessly integrated with various data sources. Then, text transformers prepare the data for processing by language models. Finally, vector storage systems index the generated embeddings, enabling efficient and precise searches.

The modularity of LangChain facilitates ongoing experimentation and evolution. Key components such as data sources, language models, and storage can be replaced or extended without the need to rewrite the entire application. This allows for flexible adaptation to new requirements and advancements in language model technology.

## Project Use Case

Within the scope of our project, we have embraced the approach of Retrieval-Augmented Generation (RAG) (LangChain, n.d.-b). This method has surfaced as a robust and efficient solution for natural language processing applications, particularly in question-answering (QA) tasks over a corpus of documents. Our implementation is grounded on the principles set by LangChain and its comprehensive documentation, which presents a modular and cohesive structure for constructing QA processing chains.

The RAG architecture unfolds across five fundamental stages:

- **Loading:** Initially, it is imperative to load the data to be analyzed. In this regard, we utilize a document loading interface provided by LangChain, which supports the integration of diverse data formats and sources.

- **Splitting:** Subsequently, the documents are fragmented into smaller pieces. This segmentation is crucial as it prepares the content for more efficient processing, facilitating its management by retrieval and generation algorithms.

- **Storage:** Once the texts are split, they are stored in a suitable structure, which, in many implementations, involves embedding text fragments with vector representations, thus facilitating semantic similarity-based retrieval.

- **Retrieval:** In this critical phase, relevant information is retrieved from the database using similarity search to identify text fragments that best align with the presented query.

- **Generation:** Lastly, a large language model (LLM) is employed to generate a relevant answer based on the question and the data retrieved in the preceding stages.

![Pipeline for converting raw unstructured data into a QA](img/pipeline.jpeg)
*Figure 1: Pipeline for converting raw unstructured data into a question-answering system (LangChain, n.d.-b)*.

In our project, we have gone a step further by modularizing each of these critical components using Python interfaces and constructors. This design strategy ensures system scalability, allowing for the easy expansion or modification of each component in response to changing requirements or technological advancements.

### Project Organization

Our project, which is hosted on the GenAI Knowledge Management with LangChain (López & Jiménez, 2023) GitHub repository, is divided into modules, with each one responsible for a specific part of the workflow, such as data extraction, embedding generation, query response, and user interface creation. This modular organization facilitates the maintainability and expansion of the project since components are interchangeable and can be adapted to different data sources and language models without costly changes to the application’s structure.

- `main.py`: The main file that initiates the application and coordinates the execution of different stages and modules.

- `models`: Directory containing the concrete implementations of the modules.

- `factories`: Directory containing the concrete implementations of the factories that create instances of the modules and configure their parameters.

- `protocols`: Directory containing the interfaces or protocols that define the contracts for the classes in the models and factories modules, following SOLID principles and allowing for greater modularity and scalability of the project.

[*View GitHub repo*](https://github.com/luisjimenezgloballogic/langchain_genai_km)

### Workflow

**Store Data:** The process begins with data extraction, carried out by the `extractor` module, which works in conjunction with `data_store` to store relevant information and with `preprocessor` to ensure data is preprocessed before being stored or used.

```python
class Application:
    ⁝       ⁝       ⁝
    def store_data(self):
        data_store: IDataStore = self.data_store_factory.create_data_store()
        data_preprocessor: IDataPreprocessor = self.data_preprocessor_factory.create_data_preprocessor()
        source_list: List[Dict] = self.extractor_factory.get_source_list()
        for source in source_list:
            extractor: IExtractor = self.extractor_factory.create_extractor(source)
            extractor.extract(data_preprocessor, data_store)
```

**Store Vectors:** Once the data is stored and preprocessed, the `loader`, `text_splitter`, and `embedding` modules come into play. The first is responsible for loading documents from different data sources, the second for dividing the documents into more manageable fragments, while the third transforms these fragments into vector representations, also known as embeddings. These embeddings are later stored by `vector_store` in a vector database, facilitating fast and efficient searches.

```python
    def store_vector(self):
        loader: BaseLoader = self.loader_factory.create_loader()
        text_splitter: TextSplitter = self.text_splitter_factory.create_text_splitter()
        embeddings: Embeddings = self.embeddings_factory.create_embeddings()
        vector_store: VectorStore = self.vector_store_factory.create_vector_store(embeddings, loader, text_splitter)
        return vector_store
```

**Chat:** When the user interacts with the system through the interface provided by `gui`, it is the `generator` module that comes into action, using `retriever` to search the vector database for the text fragments most relevant to the user's query. Subsequently, a language model, managed by `language_model`, is used to generate coherent and useful responses based on the search results.

```python
    def chat(self):
        embeddings: Embeddings = self.embeddings_factory.create_embeddings()
        vector_store: VectorStore = self.vector_store_factory.get_vector_store(embeddings)
        retriever: VectorStoreRetriever = self.retriever_factory.create_retriever(vector_store)
        llm: BaseLanguageModel = self.language_model_factory.create_language_model()
        gui: IGUI = self.gui_factory.create_gui(retriever, llm)
        generator: IGenerator = self.generator_factory.create_generator(retriever, llm)
        gui.run(generator)
```
[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/main.py#L86)

### Store Data

#### Extractor

The `extractor` module is responsible for extracting information from various sources. The `IExtractor` interface has an `extract` method that takes care of the extraction process.

The `GitHubExtractor` class implements the extraction logic specified by `IExtractor`. Its task is to navigate GitHub repositories, identify relevant files, and process their content.

```python
class GitHubExtractor(Extractor):
    ⁝       ⁝       ⁝
    def extract(self, data_preprocessor: IDataPreprocessor, data_store: IDataStore):
        ⁝       ⁝       ⁝
        self._process_directory(self.base_path)
```

The `_process_directory` method iterates over the contents of a directory in the repository and, using `_get_files`, makes calls to the GitHub API to obtain the list of files. In this process, certain files or directories may be decided to be skipped based on predefined criteria in `_skip`.

```python
    def _process_directory(self, path: str):
        if self._skip(path):
            return
        url = self._get_url(path)
        ⁝       ⁝       ⁝
        files_ok, files = self._get_files(url, self.headers)
        ⁝       ⁝       ⁝
        for file in files:
            self._process_file_or_directory(file)

    def _skip(self, path):
        # Determine if a given path should be skipped based on predefined criteria.

    def _get_url(self, path):
        # Construct the URL for accessing the GitHub repository contents at a given path.

    def _get_files(self, url, headers) -> Tuple[bool, List[Dict[str,str]]]:
        # Get a list of files from the GitHub repository using the GitHub API.
```

Files are then processed one by one in `_process_file_or_directory`, where a distinction is made between files and directories, and the content of Markdown files is specifically handled. Then the content of the file is downloaded, processed by the `data_preprocessor`, and subsequently stored with `data_store`.

```python
    def _process_file_or_directory(self, file: Dict[str, str]):
        if file["type"] == "file" and file["name"].endswith((".mdx", ".md")):
            text = self._get_file_text(file)
            if text is not None and isinstance(text, str):
                data = self.data_preprocessor.process(text)
                metadata: Dict[str, str|int] = {"owner": self.owner, "repo": self.repo, "path": file["path"]}
                self.data_store.store_data(data, metadata)
        elif file["type"] == "dir":
            self._process_directory(file["path"])

    def _get_file_text(self, file: Dict[str, str]) -> str:
        # Download the text content of a file from GitHub and return it.
```

It is observed that `GitHubExtractor` only needs to work with the interfaces `IDataPreprocessor` and `IDataStore`, which allows it to be decoupled from the concrete implementations of preprocessing and storage.

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/extractor.py#L23)

#### Data Preprocessor

The `preprocessor` module allows for the preparation and cleaning of data before it is stored. The `TextCleaner` class serves as a concrete implementation of the `IDataPreprocessor` interface and provides specific methods for processing and cleaning text.

The main function of `TextCleaner` is to process text using the `process` method. This method applies a series of transformations that include the removal of HTML tags, URLs, copyrights, emojis, and excessive whitespace, among others. The goal is to ensure that the text is in a cleaner and more uniform format for subsequent use.

```python
class TextCleaner:
    def process(self, data: str|bytes) -> str|bytes:
        # Process the text by applying a series of cleaning and normalizing transformations.
        text = self._remove_html_tags(data)
        ⁝       ⁝       ⁝
        text = self._remove_excess_whitespace(text)
        return text

    def _remove_html_tags(self, text: str) -> str:
        # Remove HTML tags from the text.
        ⁝       ⁝       ⁝
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/preprocessor.py#L8)

#### Data store

The `data_store` module provides a standardized process for storing documents, abstracting the details specific to the underlying storage system.

The `JSONLDataStore` class implements the `IDataStore` interface to handle files in the JSON Lines (JSONL) format. This format allows for the storage of structured data, where each line in the file is a separate JSON object.

The `store_data` function enables the storage of data along with associated metadata. The data and metadata are packaged into a dictionary and written to the JSONL file.

```python
class JSONLDataStore:
    ⁝       ⁝       ⁝
    def store_data(self, data: str|bytes, metadata: Dict[str, str|int]):
        file_path = self.file_system_helper.get_file_path()
        data_dict = {"metadata": metadata, "data": data}
        with open(file_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(data_dict) + "\n")
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/data_store.py#L9)

The `FileSystemHelper` class manages the creation and maintenance of the data file. It handles the file name and paths, and ensures that the data directory is prepared for storage.

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/utils/__init__.py#L14)

### Store Vectors

#### Loader

The `loader` module is responsible for loading documents from various data sources. The `JSONLLoader` class extends from the base class `BaseLoader` of LangChain and implements its `load` and `lazy_load` methods to retrieve stored data, returning them as a list (or an iterator in the case of `lazy_load`) of `Document` objects. Each line of the JSONL file is read and converted into a `Document` object, which includes both the data and the associated metadata.

```python
class JSONLLoader(BaseLoader):
    ⁝       ⁝       ⁝
    def load(self) -> List[Document]:
        file_path = self.file_system_helper.get_file_path()
        with jsonlines.open(file_path) as reader:
            return [Document(page_content=obj.get("data", ""), metadata=obj.get("metadata", {})) for obj in reader]
            
    def lazy_load(self) -> Iterator[Document]:
        file_path = self.file_system_helper.get_file_path()
        with jsonlines.open(file_path) as reader:
            return (Document(page_content=obj.get("data", ""), metadata=obj.get("metadata", {})) for obj in reader )
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/loader.py#L12)

#### Text Splitter

The `text_splitter` module segments long documents into more manageable pieces. The `ITextSplitterFactory` interface facilitates the integration of different implementations.

The specific implementation `RecursiveCharacterTextSplitter`, provided by LangChain, is responsible for breaking down the content of documents based on the size of the fragment (`chunk_size`) and the overlap between fragments (`chunk_overlap`).

```python
class RecursiveCharacterTextSplitterFactory:
    ⁝       ⁝       ⁝
    def create_text_splitter(self) -> TextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, length_function=len, chunk_overlap=self.chunk_overlap)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/text_splitter.py#L16)

#### Embedding

The `embedding` module transforms documents into vector representations, also known as embeddings. It uses the `IEmbeddingsFactory` interface to integrate with different embedding generation implementations.

`OpenAIEmbeddings`, provided by LangChain and configured with the OpenAI `text-embedding-ada-002` model, turns texts into vector representations using OpenAI models.

```python
class OpenAIEmbeddingsFactory:
    ⁝       ⁝       ⁝
    def create_embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(model=self.embedding_model)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/embedding.py#L16)

#### Vector Store

The `vector_store` module manages the vector representations of document content, known as embeddings, within the system. This module facilitates the creation, storage, and retrieval of these vectors, allowing for searches based on semantic similarity.

The `IVectorStoreFactory` interface defines the methods `create_vector_store` and `get_vector_store`. The `create_vector_store` method generates a new `VectorStore` from a list of `Documents`, a text splitter `TextSplitter`, and the `Embeddings` object. On the other hand, `get_vector_store` returns the `VectorStore` once it has been created.

The `ChromaFactory` class provides a specific implementation using LangChain's `Chroma` class to store vectors using the Anthropic Chroma library.

```python
class ChromaFactory:
    ⁝       ⁝       ⁝
    def create_vector_store(self, embeddings: Embeddings, loader: BaseLoader, text_splitter: TextSplitter) -> VectorStore:
        documents = loader.load()
        splitted_documents = text_splitter.split_documents(documents)
        return Chroma.from_documents(documents=splitted_documents, embedding=embeddings, persist_directory=self.path)

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        return Chroma(embedding_function=embeddings, persist_directory=self.path)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/vector_store.py#L19)

### Chat

#### Retriever

The `retriever` module searches for and retrieves information from a vector database based on vector representations to find the most relevant text fragments for a specific query. It relies on the `IRetrieverFactory` interface to integrate various implementations.

The specific class `NumDocsRetrieverFactory` is responsible for configuring and creating instances of `VectorStoreRetriever`, a class provided by LangChain that allows for efficient searches in vector databases, setting the number of documents (`num_docs`) that should be retrieved in each search.

```python
class NumDocsRetrieverFactory:
    ⁝       ⁝       ⁝
    def create_retriever(self, vector_db: VectorStore) -> VectorStoreRetriever:
        return vector_db.as_retriever(search_kwargs={"k": self.num_docs})
```

`VectorStoreRetriever` acts as a wrapper around `VectorStore`, leveraging its search methods, such as similarity and MMR (Maximal Marginal Relevance), to query and retrieve the most similar and relevant texts in relation to a given query.

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/retriever.py#L13)

#### Language Model

The `language_model` module generates responses based on the information retrieved and the conversation history. This module facilitates the transition from retrieving data from the vector database to synthesizing responses.

The `ILanguageModelFactory` interface provides integration with different language models. `ChatOpenAI`, configured with the `gpt-3.5-turbo` model, makes predictions by interacting with the OpenAI API to generate responses.

```python
class ChatOpenAIFactory:
    ⁝       ⁝       ⁝
    def create_language_model(self) -> BaseLanguageModel:
        return ChatOpenAI(model=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/language_model.py#L20)

#### Generator

Through the `generator` module, the user's queries are processed and transformed into answers.

The `IGenerator` interface establishes a single method, `get_answer`, which takes a query in the form of a text string and returns a response, also as a text string.

There are two concrete implementations of this interface: `QAGenerator` and `ConversationalGenerator`.

`QAGenerator` provides a direct approach, passing the query directly through the processing chain and returning the generated response, without maintaining a conversation history.

```python
class QAGenerator(Generator):
    ⁝       ⁝       ⁝
    def get_answer(self, query: str) -> str:
        answer = self.chain.run(query)
        return answer
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/generator.py#L27)

`QAGeneratorFactory` is responsible for compiling the necessary processing chain that feeds the `QAGenerator`. Using the `RetrievalQA` class from LangChain, `QAGeneratorFactory` sets up a `chain` with the language model `llm` and the document retrieval system `retriever`.

```python
class QAGeneratorFactory:
    ⁝       ⁝       ⁝
    def create_generator(self, retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> IGenerator:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type=self.chain_type, retriever=retriever)
        return QAGenerator(chain)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/generator.py#L31)

`ConversationalGenerator` is designed to maintain a history of the conversation, allowing it to generate responses that take into account the previous context. This history is stored in a list of tuples, where each tuple contains a query and the associated response. The `get_answer` method collects the query and chat history, passes them through a `chain` processing chain, and stores the query and the generated response in the history.

```python
from langchain.chains.base import Chain

class ConversationalGenerator(Generator):
    history = []
    ⁝       ⁝       ⁝
    def get_answer(self, query: str) -> str:
        result = self.chain({"question": query, "chat_history": self.history})
        answer = result["answer"]
        self.history.append((query, answer))
        return answer
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/generator.py#L20)

The creation of `chain` falls to `ConversationalGeneratorFactory`, which configures LangChain's `ConversationalRetrievalChain` with the language model `llm` and the retrieval system `retriever`. This chain is capable of condensing the conversation history and the current query to retrieve relevant documents and generate conversational responses informed by previous interactions.

```python
class ConversationalGeneratorFactory:
    ⁝       ⁝       ⁝
    def create_generator(self, retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> IGenerator:
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, verbose=self.verbose)
        return ConversationalGenerator(chain)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/generator.py#L17)

#### GUI

The `gui` module facilitates the articulation of the user's conversation with the system. This module is responsible for receiving user queries, presenting generated responses, and managing the flow of conversation. The `IGUI` interface defines an essential method, `run`, which is responsible for initiating and maintaining the interaction cycle with the user.

The `run` method takes as an argument an object that implements the `IGenerator` interface, ensuring that interaction with any concrete implementation is possible. Within the method, there is a loop that handles the user's introduction, processes the query through the `get_answer` method of the chat object, and presents the generated response.

The concrete class `CommandLineChatGUI` provides a specific implementation of `IGUI`, defining how to interact with the user through the command line. This class manages text input and output, presenting welcome messages, user prompts, and the generated responses.

```python
class CommandLineChatGUI(ChatGUI):
    ⁝       ⁝       ⁝
    def run(self, generator: IGenerator|None = None):
        generator = generator or self.generator

        self._intro()
        while True:
            self._user_message()
            query = self._get_query()

            if self._quit(query):
                break

            answer = generator.get_answer(query)
            self._ia_message(answer)
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/models/gui.py#L23)

Here, `_intro` displays the welcome messages, `_user_message` presents the prompt for the user to enter their query, `_get_query` collects and returns the user's query, `_quit` determines if the user's query indicates a desire to exit the chat, and `_ia_message` displays the generated response `answer`.

The text lines and messages displayed are configurable through the arguments passed to the initializer of `CommandLineChatGUI`, allowing for a personalized user experience.

```python
class CommandLineChatGUIFactory:
    ⁝       ⁝       ⁝
    def create_gui(self, retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> IGUI:
        return CommandLineChatGUI(generator=self._get_generator(retriever, llm),..., ia_lines=self._get_ia_lines())

    def _get_generator(self, retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> IGenerator:
        return QAGeneratorFactory(self.config_loader).create_generator(retriever=retriever, llm=llm)
    ⁝       ⁝       ⁝
    def _get_ia_lines(self):
        return ["[red]AI: [/red]"]
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/factories/gui.py#L17)

## Results

The project is configured through the `config.yaml` file, where it is established that the data source will be the documentation in the GitHub repository of LangChain (Chase, 2022).

```yaml
github:
  repos:
    - owner: langchain-ai
      repo: langchain
      path: docs/docs

loader:
  jsonl:
    file_directory: data
    file_base_name: github_data_store
    file_type: jsonl
    file_name: github_data_store_file.jsonl
    ⁝       ⁝       ⁝
```

[*View full code*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/config/config.yaml#L5)

From this repository, the file `data/github_data_store_file.jsonl` was obtained, which contains the documentation of the repository.

```json
{"metadata": {"owner": "langchain-ai", "repo": "langchain", "path": "docs/docs_skeleton/docs/guides/langsmith/README.md"}, "data": "This section has been moved to [/docs/docs/guides/langsmith]("}
{"metadata": {"owner": "langchain-ai", "repo": "langchain", "path": "docs/docs/_templates/integration.mdx"}, "data": "[comment: Please, a reference example here \"docs/integrations/arxiv.md\"]:: [comment: Use this template to create a new .md file in \"docs/integrations/\"]:: # Title_REPLACE_ME ..."}
⁝       ⁝       ⁝
```

[*View JSONL file*](https://github.com/luisjimenezgloballogic/langchain_genai_km/blob/main/data/github_data_store_file.jsonl)

With this data, the vector database was created, and the application was run using the `QAGenerator` and `ConversationalGenerator` implementations, generating two basic queries about LangChain: `what is langchain?` and `give me an example`.

[*Run App Config*](https://github.com/luisjimenezgloballogic/langchain_genai_km/tree/main/config)

Since the `gpt-3.5-turbo` model from OpenAI was used for this project, the responses were compared with those generated by OpenAI's ChatGPT 3.5, which uses the same model in its operation, to verify that the responses indeed come from the vector database and not from the language model used.

### ChatGPT 3.5

The ChatGPT 3.5 model demonstrated inherent limitations within its knowledge framework, which has a cutoff in January 2022. When queried about LangChain, the model acknowledged its incapacity to provide post-training information, illustrating one of the fundamental restrictions of language models that do not continuously update or incorporate real-time information retrieval techniques.

```
You
what is langchain?

ChatGPT
I'm not aware of any specific technology or concept called "Langchain" as of my last knowledge update in January 2022. It's possible that it may be a term or concept that has emerged after that date, or it could be a relatively niche or specialized term that is not widely recognized.

If "Langchain" is a specific term or concept that you're referring to, I would recommend providing more context or details about what it pertains to so that I can try to provide a more accurate explanation or answer.
```

### QAGenerator

Conversely, the QAGenerator, despite having access to an updated database, exhibited a lack of coherence when responding to successive questions on the same topic. This behavior highlights the absence of conversational memory, a critical component that enables AI systems to maintain coherent and contextual dialogue. Although the QAGenerator was able to offer an accurate initial description of LangChain, its inability to remember previous interactions resulted in disconnected responses, casting doubt on its efficacy for applications requiring fluid and continued interaction.

```
AI: What would you like to ask me about?

You:
what is langchain?
AI:
LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason based on the provided context. LangChain provides abstractions for working with language models and off-the-shelf chains for accomplishing specific higher-level tasks. It also provides standard interfaces and external integrations for various modules. You can install LangChain, set up your environment, and start building applications using the framework.

You:
give me an example
AI: 
Sure! Here's an example of how I can help you with everyday tasks:

User: What's the weather like today?
AI: Let me check. (AI retrieves weather information) It looks like it will be partly cloudy with a high of 75 degrees Fahrenheit.

User: Can you recommend a good restaurant nearby?
AI: Of course! What type of cuisine are you in the mood for?

User: I need help with my math homework.
AI: I'd be happy to assist you. What specific math problem are you struggling with?

User: Can you set a reminder for me to pick up groceries tomorrow?
AI: Certainly! What time would you like the reminder to be set for?

User: What's the capital of France?
AI: The capital of France is Paris.

I hope this example gives you a better idea of how I can assist you with various tasks.
```

### ConversationalGenerator

In contrast, the ConversationalGenerator demonstrated superior proficiency in terms of coherence and context retention. It maintained a consistent line of dialogue by providing detailed information on LangChain, followed by an example that aptly reflected the prior explanation. This approach is indicative of a functional conversational memory, which is essential for constructing coherent and ongoing dialogues.

```
AI: What would you like to ask me about?

You:
what is langchain?
AI: 
LangChain is a tool or framework designed to help with various natural language processing (NLP) tasks. It provides a set of functionalities and utilities for working with language models (LLMs), managing prompts, optimizing prompts, and generating text. LangChain also supports the creation of chains, which involve sequences of calls to LLMs or other utilities. It offers integrations with other tools and provides end-to-end examples for common applications such as question answering, chatbots, and agents. Additionally, LangChain supports data augmented generation, where external data sources are used to enhance the generation process. Overall, LangChain aims to simplify and streamline the development of NLP applications.

You: 
give me an example
AI: 
Yes, LangChain can be used in a natural language processing task such as question answering over specific documents. For example, you can use LangChain to build a system that can answer questions based on a given document or database. This involves using LangChain's prompt management and optimization  capabilities to formulate the question and retrieve the relevant information from the document. LangChain provides end-to-end examples and documentation on how to implement this task.
```

## Conclusions

The exploration of LangChain as a versatile framework for building smart search applications presents a compelling case for its utility in enhancing information retrieval and contextualization through language models. LangChain's modular and scalable architecture provides a solid foundation for developers to create tailored applications that can adapt and evolve with the changing needs of an organization.

The application of LangChain in our project showcases its strength in handling complex tasks such as question-answering over a corpus of documents. By leveraging the Retrieval-Augmented Generation (RAG) method, we were able to construct a pipeline that effectively transforms raw data into a coherent QA system. This was achieved by following the stages of loading, splitting, storage, retrieval, and generation, with each step carefully designed to work in tandem to process and understand natural language queries efficiently.

The project's organization into modular components allowed for a high degree of maintainability and scalability. Each module, from data extraction to user interface creation, could be developed, tested, and improved independently, providing flexibility and ease of updates.

The results of the project were particularly revealing. The `QAGenerator` and `ConversationalGenerator` each showcased unique capabilities and limitations. While the `QAGenerator` could provide accurate information on LangChain, it lacked conversational memory, resulting in a series of disjointed responses when asked follow-up questions. In contrast, the `ConversationalGenerator` demonstrated an ability to maintain context and coherence over successive interactions, showcasing the importance of conversational memory in providing an engaging and helpful user experience.

Our comparison between the `QAGenerator`'s outputs and those of ChatGPT 3.5 highlighted the advantages of having an updated database. ChatGPT 3.5 was constrained by its last knowledge update and could not provide information on LangChain post its training cut-off in January 2022. Meanwhile, our LangChain-powered generators could provide current and relevant information, demonstrating the importance of integrating real-time data retrieval capabilities in language models.

In conclusion, LangChain stands out as an adaptable framework that can significantly improve the development of NLP applications. Its ability to integrate with various language models and databases empowers developers to create sophisticated applications that are context-aware and can respond to the dynamic requirements of information retrieval tasks. The project exemplifies how the strategic use of LangChain can lead to the development of intelligent systems capable of maintaining meaningful dialogue, a critical asset in the field of AI-driven communication.

## References
- López, L., Jiménez, L.C. (2023). GenAI Knowledge Management with LangChain [Software]. https://github.com/luisjimenezgloballogic/langchain_genai_km
- LangChain. (n.d.). Chat models. LangChain Documentation. https://python.langchain.com/docs/integrations/chat/
- LangChain. (n.d.). Retrieval-augmented generation (RAG). LangChain Documentation. https://python.langchain.com/docs/use_cases/question_answering/
- Chase, H. (2022). LangChain [Software]. https://github.com/langchain-ai/langchain
- Topsakal, O., & Akinci, T. C. (2023). Creating Large Language Model Applications Utilizing LangChain: A Primer on Developing LLM Apps Fast. In International Conference on Applied Engineering and Natural Sciences (Vol. 1, No. 1, pp. 1050-1056). https://as-proceeding.com/index.php/icaens/article/view/1127/1062
- Pesaru, A., Gill, T. S., & Tangella, A. R. (2023) AI ASSISTANT FOR DOCUMENT MANAGEMENT USING LANG CHAIN AND PINECONE. https://www.irjmets.com/uploadedfiles/paper/issue_6_june_2023/42630/final/fin_irjmets1687886863.pdf