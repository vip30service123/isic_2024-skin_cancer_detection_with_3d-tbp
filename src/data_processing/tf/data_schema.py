from abc import abstractclassmethod
from typing import Tuple


class BaseDataset:
    def __init__(self, config: str) -> None:
        self.config = config

    @abstractclassmethod
    def train_test_split(self) -> Tuple:
        """
        """

    