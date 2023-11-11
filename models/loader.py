# models/loader.py

from typing import Iterator, List

import jsonlines
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

from protocols.utils import IFileSystemHelper


class JSONLLoader(BaseLoader):
    def __init__(self, file_system_helper: IFileSystemHelper):
        self.file_system_helper = file_system_helper

    def load(self) -> List[Document]:
        file_path = self.file_system_helper.get_file_path()
        with jsonlines.open(file_path) as reader:
            return [
                Document(
                    page_content=obj.get("data", ""),
                    metadata=obj.get("metadata", {}),
                )
                for obj in reader
            ]

    def lazy_load(self) -> Iterator[Document]:
        file_path = self.file_system_helper.get_file_path()
        with jsonlines.open(file_path) as reader:
            return (
                Document(
                    page_content=obj.get("data", ""),
                    metadata=obj.get("metadata", {}),
                )
                for obj in reader
            )
