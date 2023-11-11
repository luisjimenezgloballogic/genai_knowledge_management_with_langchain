# factories/loader.py

from typing import Dict

from langchain.document_loaders.base import BaseLoader

from models.loader import JSONLLoader
from protocols.config import IConfigLoader
from utils import FileSystemHelper


class JSONLLoaderFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config: Dict = config_loader.load()["loader"]["jsonl"]
        self.file_directory = self.config.get("file_directory", "data")
        self.file_base_name = self.config.get("file_base_name", "data")
        self.file_type = self.config.get("file_type", "jsonl")
        self.file_name = self.config.get("file_name", None)

    def create_loader(self) -> BaseLoader:
        file_system_helper = FileSystemHelper(
            file_directory=self.file_directory,
            file_base_name=self.file_base_name,
            file_name=self.file_name,
        )
        return JSONLLoader(file_system_helper=file_system_helper)
