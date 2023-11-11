# ====== config module =======

# utils/__init__.py

import datetime
import os

from dotenv import load_dotenv
from termcolor import colored

load_dotenv()


class FileSystemHelper:
    def __init__(
        self,
        file_directory: str = "directory",
        file_base_name: str = "base_name",
        file_type: str = "txt",
        file_name: str|None = None,
    ):
        self._file_directory = file_directory
        self._file_base_name = file_base_name
        self._file_type = file_type
        self._file_name = file_name

    def create_dir(self) -> None:
        dir_rel_path = self._file_directory
        dir_rel_path = os.path.normpath(dir_rel_path)
        if not os.path.exists(dir_rel_path):
            os.makedirs(dir_rel_path)

    def remove_existing_file(self) -> None:
        file_path: str = self.get_file_path()
        file_path = os.path.normpath(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)

    def get_file_path(self) -> str:
        dir_path: str = self._file_directory
        file_name: str = self.get_file_name()
        file_rel_path = os.path.join(dir_path, file_name)
        file_rel_path = os.path.normpath(file_rel_path)
        return file_rel_path

    @staticmethod
    def get_abs_path(rel_path: str) -> str:
        working_dir = os.getcwd()
        abs_path = os.path.join(working_dir, rel_path)
        abs_path = os.path.normpath(abs_path)
        return abs_path

    def get_file_name(self) -> str:
        if self._file_name:
            return self._file_name
        current_date = datetime.date.today().strftime("%Y_%m_%d")
        file_name = f"{self._file_base_name}_{current_date}.{self._file_type}"
        return file_name


class ColorLogger:
    @staticmethod
    def info(message: str):
        print(colored(message, "blue"))

    @staticmethod
    def success(message: str):
        print(colored(message, "green"))

    @staticmethod
    def warning(message: str):
        print(colored(message, "yellow"))

    @staticmethod
    def error(message: str):
        print(colored(message, "red"))
