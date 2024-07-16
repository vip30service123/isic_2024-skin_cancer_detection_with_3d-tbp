
from omegaconf import DictConfig
import torch
from torch import nn
import torchvision


class Resnet50(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.model = torchvision.models.resnet50()
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2048,
                out_features=2
            ),
            torch.nn.Sigmoid()
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor, label: torch.Tensor = None):
        output = self.model(image)
        if type(label) != None:
            loss = self.loss_fn(output, label)
            return {
                "logits": output,
                "loss": loss
            }
        else:
            return {
                "logits": output
            }