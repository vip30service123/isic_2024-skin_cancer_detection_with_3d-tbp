from tqdm.auto import tqdm

from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim 



class Trainer:
    def __init__(
        self,
        train_dl: DataLoader,
        model: nn.Module,
        optimizer: optim,
        device: str = 'cpu',
        validate_dl: DataLoader = None,
        epochs: int = 3,
    ):
        self.train_dl = train_dl
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.validate_dl = validate_dl
        self.epochs = epochs


    @staticmethod
    def _run_one(
        dataloader: DataLoader, 
        model: nn.Module, 
        device: str = 'cpu', 
        optimizer: optim = None, 
    ) -> float:
        """Calculate avarage loss when forwarding all dataloader items.
        
        Args:
            dataloader: torch dataloader.
            model: nn.Module.
            device: cuda or cpu.
            optimizer: to train model parameter. Will not train if it's None.

        Returns:
            Average loss.
        """

        assert type(dataloader) == DataLoader, f"dataset must be of type torch.utils.data.Dataset, not {type(dataloader)}"
        assert type(device) == str, f"device must be of type str, not {type(device)}"

        total_loss: float  = 0
        
        if optimizer is not None:
            model.train()

            for batch in tqdm(dataloader, desc="training"):
                batch = {i: j.to(device) for i, j in batch.items()}
                loss = model(**batch)['loss']
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="validating"):
                    batch = {i: j.to(device) for i, j in batch.items()}
                    total_loss += model(**batch)['loss'].item()

        model.eval()

        average_loss = total_loss / len(dataloader)

        return average_loss



    def train(self) -> None:  
        """Train model."""
        
        assert type(self.train_dl) == DataLoader, f"train_ds must be of type torch.utils.data.DataLoader, not {type(self.train_dl)}"
        assert type(self.validate_dl) == DataLoader or type(self.validate_dl) == None, f"valivate_ds must be of type torch.utils.data.DataLoader or None, not {type(self.validate_dl)}"
        assert type(self.device) == str, f"device must be of type str, not {type(self.device)}"

        for epoch in tqdm(range(self.epochs)):
            print(f"#### Epoch {epoch}")
            train_loss = self._run_one(
                dataloader=self.train_dl, 
                model=self.model, 
                device=self.device,
                optimizer=self.optimizer
            )
            print(f"Train loss: {train_loss}")

            if self.validate_dl is not None:
                validate_loss = self._run_one(
                    dataloader=self.validate_dl,
                    model=self.model,
                    device=self.device
                )
                print(f"Validate loss: {validate_loss}")
        

