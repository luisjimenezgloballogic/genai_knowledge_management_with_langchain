# ====== extractor module =======

# protocols/models/extractor.py

from typing import Protocol

from protocols.models.data_store import IDataStore
from protocols.models.preprocessor import IDataPreprocessor


class IExtractor(Protocol):
    def extract(self, data_preprocessor: IDataPreprocessor, data_store: IDataStore):
        ...
