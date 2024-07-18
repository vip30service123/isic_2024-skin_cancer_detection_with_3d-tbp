import io
from typing import Any, List

import h5py
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, isic_ids: List[str], targets: List[int], config: DictConfig, transform: torchvision.transforms.Compose):
        self.isic_ids = isic_ids
        self.config = config
        self.targets = targets
        self.transform = transform


    def __len__(self) -> int:
        return len(self.isic_ids)


    def __getitem__(self, id: int) -> Any:
        item_id = self.isic_ids[id]
        target = self.targets[id]
        target = [1, 0] if target == 0 else [0, 1]
        target = torch.Tensor(target)

        with h5py.File(self.config['dataset']['directory'], "r") as f:
            im = Image.open(io.BytesIO(f[item_id][()]))

            pix = np.array(im)

        pix = self.transform(pix)

        return {
            "image": pix, 
            "label": target
        }
