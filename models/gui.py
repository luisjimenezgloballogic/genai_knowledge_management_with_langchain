# models/gui.py

from abc import ABC, abstractmethod
from typing import List

from rich.console import Console

from protocols.models.generator import IGenerator

console = Console()


class GUI(ABC):
    def __init__(self, generator: IGenerator) -> None:
        self.generator = generator

    @abstractmethod
    def run(self, generator: IGenerator|None = None):
        pass


class ChatGUI(GUI):
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

    @abstractmethod
    def _intro(self):
        pass

    @abstractmethod
    def _user_message(self):
        pass

    @abstractmethod
    def _get_query(self) -> str:
        pass

    @abstractmethod
    def _quit(self, query: str) -> bool:
        pass

    @abstractmethod
    def _ia_message(self, answer: str):
        pass


class CommandLineChatGUI(ChatGUI):
    query: str

    def __init__(
        self,
        generator: IGenerator,
        intro_lines: List[str],
        user_lines: List[str],
        query_error_message: str,
        quit_word_list: List[str],
        ia_lines: List[str],
    ) -> None:
        super().__init__(generator)
        self.intro_lines = intro_lines
        self.user_lines = user_lines
        self.query_error_message = query_error_message
        self.quit_word_list = quit_word_list
        self.ia_lines = ia_lines

    def _intro(self):
        for line in self.intro_lines:
            console.print(line)

    def _user_message(self):
        console.print("")
        for line in self.user_lines:
            console.print(line)

    def _get_query(self) -> str:
        try:
            self.query = input()
            return self.query
        except EOFError:
            print(self.query_error_message)
            return self._get_query()

    def _quit(self, query: str):
        return query in self.quit_word_list

    def _ia_message(self, answer: str):
        for line in self.ia_lines:
            console.print(line)
        console.print(answer)
