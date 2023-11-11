# factories/extractor.py

from typing import Dict, List

from config import GitHubTokenLoader
from models.extractor import GitHubExtractor
from protocols.config import IConfigLoader
from protocols.models.extractor import IExtractor


class GitHubExtractorFactory:
    def __init__(self, config_loader: IConfigLoader):
        config_loader = GitHubTokenLoader(config_loader)
        self.config = config_loader.load()
        self.github_repos: List[Dict] = self.config["github"]["repos"]
        self.headers = self.config["github_headers"]

    def create_extractor(self, source: Dict) -> IExtractor:
        return GitHubExtractor(
            owner=source["owner"],
            repo=source["repo"],
            path=source["path"],
            headers=self.headers,
        )

    def get_source_list(self) -> List[Dict]:
        return self.github_repos
