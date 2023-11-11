# models/extractor.py

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import requests

from protocols.models.data_store import IDataStore
from protocols.models.preprocessor import IDataPreprocessor
from utils import ColorLogger


class Extractor(ABC):
    data_store: IDataStore
    data_preprocessor: IDataPreprocessor

    @abstractmethod
    def extract(self, data_preprocessor: IDataPreprocessor, data_store: IDataStore):
        pass


class GitHubExtractor(Extractor):
    def __init__(self, owner: str, repo: str, path: str, headers: Dict):
        self.owner = owner
        self.repo = repo
        self.base_path = path
        self.headers = headers

    def extract(self, data_preprocessor: IDataPreprocessor, data_store: IDataStore):
        self.data_preprocessor = data_preprocessor
        self.data_store = data_store
        self._process_directory(self.base_path)

    def _process_directory(self, path: str):
        if self._skip(path):
            return
        url = self._get_url(path)
        ColorLogger.info(f"Processing directory: {path} from repository: {self.repo}")
        files_ok, files = self._get_files(url, self.headers)
        if not files_ok:
            ColorLogger.error(
                "Failed to retrieve files. Please check your GitHub token and repository details."
            )
            return

        for file in files:
            self._process_file_or_directory(file)

    def _skip(self, path):
        # Skip Chinese translations
        if os.path.basename(path) == "zh":
            return True
        return False

    def _get_url(self, path):
        base_url = (
            f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{path}"
        )
        return base_url

    def _get_files(self, url, headers) -> Tuple[bool, List[Dict[str,str]]]:
        response = requests.get(url, headers=headers)
        files_ok = response.status_code == 200

        if files_ok:
            files = response.json()
            return files_ok, files

        return files_ok, [{"":""}]

    def _process_file_or_directory(self, file: Dict[str, str]):
        if file["type"] == "file" and file["name"].endswith((".mdx", ".md")):
            text = self._get_file_text(file)
            if text is not None and isinstance(text, str):
                data = self.data_preprocessor.process(text)
                metadata: Dict[str, str|int] = {
                    "owner": self.owner,
                    "repo": self.repo,
                    "path": file["path"],
                }
                self.data_store.store_data(data, metadata)
        elif file["type"] == "dir":
            self._process_directory(file["path"])

    def _get_file_text(self, file: Dict[str, str]) -> str:
        ColorLogger.success(f"Downloading document: {file['name']}")
        ColorLogger.info(f"Download URL: {file['download_url']}")
        response = requests.get(file["download_url"])
        if response.status_code == 200:
            return response.text
        else:
            error_message = f"Failed to download the file: {file['name']}"
            ColorLogger.error(error_message)
            return error_message
