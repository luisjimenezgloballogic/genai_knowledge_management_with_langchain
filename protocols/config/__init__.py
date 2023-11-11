# ====== config module =======

# protocols/config/__init__.py

from typing import Dict, Protocol


class IConfigLoader(Protocol):
    def load(self) -> Dict:
        ...
