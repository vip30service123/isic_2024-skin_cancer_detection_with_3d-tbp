"""
- Dataset seems fixed
- Should specify model for scaling
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime

import hydra
from hydra.utils import instantiate
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

from src.dataset.tf.dataset import DatasetFromGenerator
from src.models.tf.resnet50 import Resnet50 as TFResnet50
from src.models.torch.resnet50 import Resnet50 as TorchResnet50
from src.dataset.dataset_processing import *
from src.dataset.torch.dataset import CustomDataset
from src.models.torch.trainer import trainer


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    df = GetTrainMetadataDF()(config)
    df = GetMainAttribute()(df)
    df = Augmentating(config['meta_data']['augmentation_strategy'])(df, config)
    df = SuffleDataset()(df, config)

    train_df, test_df = TrainTestSplit()(df, config)

    if 'model_type' in config:
        model_type = config['model_type']
    else:
        model_type = 'tf'

    if model_type == "tf":
        train_ds_len = config['dataset']['train_ds_len']
        test_ds_len = config['dataset']['test_ds_len']

        if not train_ds_len and not test_ds_len:
            train_ds = DatasetFromGenerator(train_df['isic_id'].tolist(), train_df['target'].tolist(), config).repeat()
            test_ds = DatasetFromGenerator(test_df['isic_id'].tolist(), test_df['target'].tolist(), config).repeat()
        elif train_ds_len and test_ds_len:
            train_ds = DatasetFromGenerator(train_df['isic_id'].tolist()[:train_ds_len], train_df['target'].tolist()[:train_ds_len], config).repeat()
            test_ds = DatasetFromGenerator(test_df['isic_id'].tolist()[:test_ds_len], test_df['target'].tolist()[:test_ds_len], config).repeat()
        else:
            raise Exception("Either full train or train small size on both train and test dataset.")

        batch_sz = config['dataset']['batch_size']
        train_ds = train_ds.batch(batch_sz)
        test_ds = test_ds.batch(batch_sz)

        model = TFResnet50()(config, show_summary=True)

        epochs=config['training']['epochs']
        steps_per_epoch = train_ds_len//batch_sz
        validation_steps = test_ds_len//batch_sz

        model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

        save_model_path = config['model']['save_model_path']
        model.save(os.path.join(save_model_path, "first.keras"))

    elif model_type == "torch":
        transform_pipe = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                size=(224, 224)
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_ds_len = config['dataset']['train_ds_len']
        test_ds_len = config['dataset']['test_ds_len']

        if not train_ds_len and not test_ds_len:
            train_ds = CustomDataset(
                train_df['isic_id'].tolist(), 
                train_df['target'].tolist(), 
                config,
                transform=transform_pipe)
            test_ds = CustomDataset(
                test_df['isic_id'].tolist(), 
                test_df['target'].tolist(), 
                config,
                transform=transform_pipe)
        elif train_ds_len and test_ds_len:
            train_ds = CustomDataset(
                train_df['isic_id'].tolist()[:train_ds_len], 
                train_df['target'].tolist()[:train_ds_len], 
                config,
                transform=transform_pipe)
            test_ds = CustomDataset(
                test_df['isic_id'].tolist()[:test_ds_len], 
                test_df['target'].tolist()[:test_ds_len], 
                config,
                transform=transform_pipe)
        else:
            raise Exception("Either full train or train small size on both train and test dataset.")
        
        batch_sz = config['dataset']['batch_size']
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_sz
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_sz
        )

        model = instantiate(config['model']['model'])

        partial_optimizer = instantiate(config['training']['optimizer'])
        optimizer = partial_optimizer(params=model.parameters())

        epochs = config['training']['epochs']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device: ", device)

        model.to(device)

        trainer(
            train_dl=train_dl,
            model=model,
            optimizer=optimizer,
            validate_dl=test_dl,
            epochs=epochs,
            device=device
        )

        time = str(datetime.datetime.now()).replace(' ', '-')

        model_save_path = config['model']['save_model_path'] + "_" + time + ".pt"

        torch.save(model, model_save_path)

if __name__=="__main__":
    main()
