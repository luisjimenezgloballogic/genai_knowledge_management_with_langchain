# factories/data_store.py

from typing import Dict

from models.data_store import JSONLDataStore
from protocols.config import IConfigLoader
from protocols.models.data_store import IDataStore
from utils import FileSystemHelper


class JSONLDataStoreFactory:
    def __init__(self, config_loader: IConfigLoader):
        self.config: Dict = config_loader.load()["loader"]["jsonl"]
        self.file_directory = self.config.get("file_directory", "data")
        self.file_base_name = self.config.get("file_base_name", "data")
        self.file_type = self.config.get("file_type", "jsonl")
        self.file_name = self.config.get("file_name", None)

    def create_data_store(self) -> IDataStore:
        file_system_helper = FileSystemHelper(
            file_directory=self.file_directory,
            file_base_name=self.file_base_name,
            file_name=self.file_name,
        )
        return JSONLDataStore(
            file_system_helper=file_system_helper,
            overwrite=True,
        )
