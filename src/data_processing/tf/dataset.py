from typing import Self

from omegaconf import DictConfig
import tensorflow as tf

from data_schema import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, config: DictConfig) -> Self:
        self.config = config

    def train_test_split(self, config: DictConfig):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=config['directory'],
            labels=config['labels'],
            label_mode=config['label_mode'],
            class_names=config['class_name'],
            color_mode=config['color_mode'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            shuffle=config['shuffle'],
            seed=config['seed'],
            validation_split=config['validation_split'],
            subset="training",
            interpolation=config['interpolation'],
            follow_links=config['follow_links'],
            crop_to_aspect_ratio=config['crop_to_aspect_ratio'],
            pad_to_aspect_ratio=config['pad_to_aspect_ratio'],
            data_format=config['data_format'],
            verbose=config['verbose']
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=config['directory'],
            labels=config['labels'],
            label_mode=config['label_mode'],
            class_names=config['class_name'],
            color_mode=config['color_mode'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            shuffle=config['shuffle'],
            seed=config['seed'],
            validation_split=config['validation_split'],
            subset="validation",
            interpolation=config['interpolation'],
            follow_links=config['follow_links'],
            crop_to_aspect_ratio=config['crop_to_aspect_ratio'],
            pad_to_aspect_ratio=config['pad_to_aspect_ratio'],
            data_format=config['data_format'],
            verbose=config['verbose']
        )

        return train_ds, val_ds

    