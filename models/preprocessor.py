# models/preprocessor.py

import re

from emoji import demojize


class TextCleaner:
    def process(self, data: str|bytes) -> str|bytes:
        if not isinstance(data, str):
            raise TypeError("The 'data' parameter must be of type 'str'.")
        
        text = self._remove_html_tags(data)
        text = self._remove_urls(text)
        text = self._remove_copyright(text)
        text = self._replace_newlines(text)
        text = self._demojize(text)
        text = self._remove_excess_whitespace(text)
        return text

    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r"<[^>]*>", "", text)

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www.\S+", "", text)

    def _remove_copyright(self, text: str) -> str:
        return re.sub(r"Copyright.*", "", text)

    def _replace_newlines(self, text: str) -> str:
        return text.replace("\n", " ")

    def _demojize(self, text: str) -> str:
        text = demojize(text)
        return re.sub(r":[a-z_&+-]+:", "", text)

    def _remove_excess_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text
