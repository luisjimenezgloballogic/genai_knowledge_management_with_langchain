# protocols/factories/preprocessor.py

from typing import Protocol

from protocols.models.preprocessor import IDataPreprocessor


class IDataPreprocessorFactory(Protocol):
    def create_data_preprocessor(self) -> IDataPreprocessor:
        ...
