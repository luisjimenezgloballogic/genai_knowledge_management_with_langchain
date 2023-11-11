# factories/language_model.py

from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel

from config import OpenAIAPIKeyLoader
from protocols.config import IConfigLoader


class ChatOpenAIFactory:
    def __init__(self, config_loader: IConfigLoader):
        config_loader = OpenAIAPIKeyLoader(config_loader)
        self.config: Dict = config_loader.load()['chat']
        self.model_name = self.config["model_name"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]

    def create_language_model(self) -> BaseLanguageModel:
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
