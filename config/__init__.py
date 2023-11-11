# ====== config module =======

# config/__init__.py

import os
import sys
from typing import Dict

import yaml
from dotenv import load_dotenv

from protocols.config import IConfigLoader


class YamlConfigLoader:
    def __init__(self, config_file_name="config.yaml"):
        self.CONFIG_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_file_name
        )

    def load(self) -> Dict:
        with open(self.CONFIG_PATH) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return {"error": exc}


class GitHubTokenLoader:
    def __init__(
        self, config_loader: IConfigLoader, github_token_env_var_name="GITHUB_TOKEN"
    ):
        self._github_token_env_var_name = github_token_env_var_name
        self._config_loader = config_loader

    def load(self) -> Dict:
        config: Dict = self._config_loader.load()
        self._token = self._get_github_token()
        config["github_token"] = self._token
        config["github_headers"] = self._get_headers(self._token)
        return config

    def _get_github_token(self):
        load_dotenv()
        token = os.getenv(self._github_token_env_var_name)
        if not token:
            raise ValueError(
                f"{self._github_token_env_var_name} no está configurado en las variables de entorno."
            )
        return token

    def _get_headers(self, token):
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3.raw",
        }


class OpenAIAPIKeyLoader:
    def __init__(
        self, config_loader: IConfigLoader, openai_api_key_env_var_name="OPENAI_API_KEY"
    ):
        self._openai_api_key_env_var_name = openai_api_key_env_var_name
        self._config_loader = config_loader

    def load(self) -> Dict:
        config: Dict = self._config_loader.load()
        config["openai_api_key"] = self._get_openai_key()
        return config

    def _get_openai_key(self):
        load_dotenv()
        key = os.getenv(self._openai_api_key_env_var_name)
        if not key:
            print("Por favor crea una variable de ambiente OPENAI_API_KEY.")
            sys.exit()
        return key


class CohereAPIKeyProvider:
    def __init__(
        self, config_loader: IConfigLoader, cohere_api_key_env_var_name="COHERE_API_KEY"
    ):
        self._cohere_api_key_env_var_name = cohere_api_key_env_var_name
        self._config_loader = config_loader

    def load(self) -> Dict:
        config: Dict = self._config_loader.load()
        config["cohere_api_key"] = self._get_cohere_key()
        return config

    def _get_cohere_key(self) -> str:
        load_dotenv()
        key = os.getenv(self._cohere_api_key_env_var_name)
        if not key:
            return input("Por favor ingresa tu COHERE_API_KEY: ")
        return key
