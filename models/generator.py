# models/generator.py

from abc import ABC, abstractmethod

from langchain.chains.base import Chain


class Generator(ABC):
    def __init__(self, chain: Chain):
        self.chain = chain

    @abstractmethod
    def get_answer(self, query: str) -> str:
        pass


class ConversationalGenerator(Generator):
    history = []

    def get_answer(self, query: str) -> str:
        result = self.chain({"question": query, "chat_history": self.history})
        answer = result["answer"]
        self.history.append((query, answer))
        return answer


class QAGenerator(Generator):
    def get_answer(self, query: str) -> str:
        answer = self.chain.run(query)
        return answer
