from typing import Self, Union

from omegaconf import DictConfig

from .simple_model import BaseModel



class SimpleModel(BaseModel):
    def __init__(self, config: DictConfig) -> Self:
        self.config = config
        self._model = _instantiate_model()
    
    def _instantiate_model(self) -> Sequential:
        pass

    def save_model(self, path: str) -> None:
        pass

    def load_model(self, path: str) -> None:
        pass

    def forward(self, input_path: str) -> Union[int, float]:
        pass

    def train(self) -> None:
        pass

    def get_model(self) -> Sequential:
        return self._model