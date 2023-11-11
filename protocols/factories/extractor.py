# protocols/factories/extractor.py

from typing import Dict, List, Protocol

from protocols.models.extractor import IExtractor


class IExtractorFactory(Protocol):
    def create_extractor(self, source: Dict) -> IExtractor:
        ...

    def get_source_list(self) -> List[Dict]:
        ...
