# protocols/factories/data_store.py

from typing import Protocol

from protocols.models.data_store import IDataStore


class IDataStoreFactory(Protocol):
    def create_data_store(self) -> IDataStore:
        ...
