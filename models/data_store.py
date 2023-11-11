# models/data_store.py

import json
from typing import Dict

from protocols.utils import IFileSystemHelper


class JSONLDataStore:
    def __init__(self, file_system_helper: IFileSystemHelper, overwrite=True):
        self.file_system_helper = file_system_helper
        self.file_system_helper.create_dir()
        if overwrite:
            self.file_system_helper.remove_existing_file()

    def store_data(self, data: str|bytes, metadata: Dict[str, str|int]):
        file_path = self.file_system_helper.get_file_path()
        data_dict = {"metadata": metadata, "data": data}
        with open(file_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(data_dict) + "\n")
