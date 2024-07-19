


from typing import Dict

import torch
from torch import nn
import torchvision


class EfficientNet(nn.Module):
    def __init__(self, num_classes: int = 2, model_type: str = "b0"):
        assert type(num_classes) == int, f"num_classes type must be int, not {type(num_classes)}"
        assert model_type in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], f"model_type must be b0, ..., b7"

        super().__init__()

        if model_type == 'b0':
            self.model = torchvision.models.efficientnet_b0()
        elif model_type == 'b1':
            self.model = torchvision.models.efficientnet_b1()
        elif model_type == 'b2':
            self.model = torchvision.models.efficientnet_b2()
        elif model_type == 'b3':
            self.model = torchvision.models.efficientnet_b3()
        elif model_type == 'b4':
            self.model = torchvision.models.efficientnet_b4()
        elif model_type == 'b5':
            self.model = torchvision.models.efficientnet_b5()
        elif model_type == 'b6':
            self.model = torchvision.models.efficientnet_b6()
        elif model_type == 'b7':
            self.model = torchvision.models.efficientnet_b7()
        
        self.model.classifier = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=1280,
                out_features=num_classes
            ),
            nn.Sigmoid()
        )
        self.loss_fn = nn.CrossEntropyLoss()

    
    def forward(self, image: torch.Tensor, label: torch.Tensor) -> Dict:
        output = self.model(image)
        loss = None

        if label is not None:
            loss = self.loss_fn(output, label)
        
        return {
            "logits": output,
            "loss": loss
        }