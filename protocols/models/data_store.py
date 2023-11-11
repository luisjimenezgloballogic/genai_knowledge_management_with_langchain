# ====== data_store module =======

# protocols/models/data_store.py

from typing import Dict, Protocol


class IDataStore(Protocol):
    def store_data(self, data: str|bytes, metadata: Dict[str, str|int]):
        ...
