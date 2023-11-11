# ====== config module =======

# protocols/utils/__init__.py

from typing import Protocol


class IFileSystemHelper(Protocol):
    def get_file_path(self) -> str:
        ...

    def create_dir(self) -> None:
        ...

    def remove_existing_file(self) -> None:
        ...
