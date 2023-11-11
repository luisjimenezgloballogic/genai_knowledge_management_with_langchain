# ====== preprocessor module =======

# protocols/models/preprocessor.py

from typing import Protocol


class IDataPreprocessor(Protocol):
    def process(self, data: str|bytes) -> str|bytes:
        ...
