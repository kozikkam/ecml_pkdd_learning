from abc import ABC, abstractmethod


class DataGenerator(ABC):
    def __init__(self, data_file_path: str):
        self._data_file_path = data_file_path

    def __iter__(self):
        return self._get_iterator()

    @abstractmethod
    def _get_iterator(self):
        pass