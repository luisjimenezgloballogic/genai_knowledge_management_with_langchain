# factories/preprocessor.py

from models.preprocessor import TextCleaner
from protocols.models.preprocessor import IDataPreprocessor


class TextCleanerFactory:
    def create_data_preprocessor(self) -> IDataPreprocessor:
        return TextCleaner()
