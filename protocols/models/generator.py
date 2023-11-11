# ====== generator module =======

# protocols/models/generator.py

from typing import Protocol


class IGenerator(Protocol):
    def get_answer(self, query: str) -> str:
        ...
