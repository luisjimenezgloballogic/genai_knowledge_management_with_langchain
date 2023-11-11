# ====== gui module =======

# protocols/models/gui.py

from typing import Protocol

from protocols.models.generator import IGenerator


class IGUI(Protocol):
    def run(self, generator: IGenerator|None = None):
        ...
