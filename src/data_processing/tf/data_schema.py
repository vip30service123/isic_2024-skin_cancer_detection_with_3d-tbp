from abc import abstractclassmethod
from typing import Self






class BaseDataset:
    def __init__(self, config: str) -> Self:
        self.config = config

    @abstractclassmethod
    def train_test_split(self) -> Tuple:
        """
        """

    