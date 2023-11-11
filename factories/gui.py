# factories/gui.py

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStoreRetriever

from factories.generator import ConversationalGeneratorFactory, QAGeneratorFactory
from models.gui import CommandLineChatGUI
from protocols.config import IConfigLoader
from protocols.models.generator import IGenerator
from protocols.models.gui import IGUI


class CommandLineChatGUIFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config_loader = config_loader

    def create_gui(
        self, retriever: VectorStoreRetriever, llm: BaseLanguageModel
    ) -> IGUI:
        return CommandLineChatGUI(
            generator=self._get_generator(retriever, llm),
            intro_lines=self._get_intro_lines(),
            user_lines=self._get_user_lines(),
            query_error_message=self._get_query_error_message(),
            quit_word_list=self._get_quit_word_list(),
            ia_lines=self._get_ia_lines(),
        )

    def _get_generator(
        self, retriever: VectorStoreRetriever, llm: BaseLanguageModel
    ) -> IGenerator:
        return QAGeneratorFactory(self.config_loader).create_generator(
            retriever=retriever, llm=llm
        )

    def _get_intro_lines(self):
        return ["[blue]AI: [/blue] What would you like to ask me about?"]

    def _get_user_lines(self):
        return ["[blue]You: [/blue]"]

    def _get_query_error_message(self):
        return "Error: Unexpected input. Please try again."

    def _get_quit_word_list(self):
        return ["leave", "quit", "exit"]

    def _get_ia_lines(self):
        return ["[red]AI: [/red]"]


class StatelessCommandLineChatGUIFactory(CommandLineChatGUIFactory):
    def _get_intro_lines(self):
        return [
            "[green]Question and Answer Mode (stateless).[/green]",
            "[blue]AI: [/blue] What would you like to ask me about?",
        ]


class MemoryCommandLineChatGUIFactory(CommandLineChatGUIFactory):
    def _get_generator(
        self, retriever: VectorStoreRetriever, llm: BaseLanguageModel
    ) -> IGenerator:
        return ConversationalGeneratorFactory(self.config_loader).create_generator(
            retriever=retriever, llm=llm
        )

    def _get_intro_lines(self):
        return [
            "[green]Memory Mode.[/green]",
            "[blue]AI: [/blue] What would you like to ask me about?",
        ]
