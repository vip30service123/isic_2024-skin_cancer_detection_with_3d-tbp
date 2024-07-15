import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math

import hydra
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import torch
from torch import nn
import torchvision

from src.data_processing.tf.dataset import Dataset
from src.models.tf.resnet50 import Resnet50
from src.data_processing.tf.dataset_processing import DatasetProcessor

# tf.keras.config.disable_traceback_filtering()


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    model_type = "torch"

    if model_type == "tf":
        data_processor = DatasetProcessor()
        df = data_processor.get_train_metadata_df(config)
        df = data_processor.get_id_target_columns(df)
        df = data_processor.augmentation(df, config)
        df = data_processor.shuffle_df(df, config)

        train_df, test_df = data_processor.train_test_split(df, config)

        train_ds_len = config['dataset']['train_ds_len']
        test_ds_len = config['dataset']['test_ds_len']

        train_ds = data_processor.dataset_from_generator(train_df['isic_id'].tolist()[:train_ds_len], train_df['target'].tolist()[:train_ds_len], config).repeat()
        test_ds = data_processor.dataset_from_generator(test_df['isic_id'].tolist()[:test_ds_len], test_df['target'].tolist()[:test_ds_len], config).repeat()

        batch_sz = config['dataset']['batch_size']
        train_ds = train_ds.batch(batch_sz)
        test_ds = test_ds.batch(batch_sz)

        model = Resnet50.from_config(config, show_summary=True)

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
        model = torchvision.models.resnet50(out_features=1000)
        print(model)


if __name__=="__main__":
    main()
