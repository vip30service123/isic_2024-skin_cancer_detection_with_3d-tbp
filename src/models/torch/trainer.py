from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim 


def run_one(
    dataloader: DataLoader, 
    model: nn.Module, 
    device: str = 'cpu', 
    optimizer: optim = None, 
) -> float:
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



def trainer(
    train_dl: DataLoader,
    model: nn.Module,
    optimizer: optim,
    device: str = 'cpu',
    validate_dl: DataLoader = None,
    epochs: int = 3,
) -> None:  
    
    assert type(train_dl) == DataLoader, f"train_ds must be of type torch.utils.data.DataLoader, not {type(train_dl)}"
    assert type(validate_dl) == DataLoader or type(validate_dl) == None, f"valivate_ds must be of type torch.utils.data.DataLoader or None, not {type(validate_dl)}"
    assert type(device) == str, f"device must be of type str, not {type(device)}"

    for epoch in tqdm(range(epochs)):
        print(f"#### Epoch {epoch}")
        train_loss = run_one(
            dataloader=train_dl, 
            model=model, 
            device=device,
            optimizer=optimizer
        )
        print(f"Train loss: {train_loss}")

        if validate_dl is not None:
            validate_loss = run_one(
                dataloader=validate_dl,
                model=model,
                device=device
            )
            print(f"Validate loss: {validate_loss}")
    

