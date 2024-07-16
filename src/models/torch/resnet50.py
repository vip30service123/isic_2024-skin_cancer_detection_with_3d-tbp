from typing import Dict

from omegaconf import DictConfig
import torch
from torch import nn
import torchvision


class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50()
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2048,
                out_features=2
            ),
            torch.nn.Sigmoid()
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor, label: torch.Tensor = None) -> Dict:
        output = self.model(image)
        loss = None
        if label is not None:
            loss = self.loss_fn(output, label)
        
        return {
            "logits": output,
            "loss": loss
        }