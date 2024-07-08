from abc import abstractmethod
from typing import Union

from omegaconf import DictConfig
from tensorflow.keras.models import Sequential


class BaseModel:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.model = _instantiate_model()

    @abstractmethod
    def _instantiate_model(self) -> Sequential:
        """
        """

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        """

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        """

    @abstractmethod
    def forward(self, input_path: str) -> Union[int, float]:
        """
        """
    
    @abstractmethod
    def train(self) -> None:
        """
        """

    @abstractmethod
    def get_model(self) -> Sequential:
        """
        """