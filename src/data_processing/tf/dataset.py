"""
Outdated,just put here, not use
"""


import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
import random
from tqdm.auto import tqdm
from typing import Optional, Tuple

import h5py
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import tensorflow as tf

from src.data_processing.tf.data_schema import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self._instantiate()


    def _instantiate(self):
        self.train_metadata = pd.read_csv(self.config['meta_data']['train_meta_data_path'])
        self.test_metadata = pd.read_csv(self.config['meta_data']['test_meta_data_path'])
        self.dataset = None


    def train_test_split(self) -> Tuple:
        train_ratio, val_ratio, test_radio = self.config['dataset']['train_val_test_split']
        dataset_len = 1000
        train_sz = int(dataset_len * train_ratio)
        val_sz = int(dataset_len * val_ratio)

        self.dataset = self.dataset.shuffle(10, seed=self.config['dataset']['seed'])
        train_ds = self.dataset.take(train_sz).batch(self.config['dataset']['batch_size'])
        val_ds = self.dataset.skip(train_sz).take(val_sz).batch(self.config['dataset']['batch_size'])
        test_ds = self.dataset.skip(train_sz + val_sz).batch(self.config['dataset']['batch_size'])

        return train_ds, val_ds, test_ds


    def prepare_dataset(self) -> None:
        labels = self.train_metadata['target'].copy().tolist()[:1000]
        isic_ids = self.train_metadata['isic_id'].copy().tolist()[:1000]

        with h5py.File(self.config['dataset']['directory']) as f:
            data = []
            for id in tqdm(isic_ids, desc="Image"):
                im = Image.open(io.BytesIO(f[id][()]))

                h, w, c = self.config['model']['input_shape']
                im = im.resize((h, w))
                pix = np.array(im)
                data.append(pix)
                del im, pix
            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        self.dataset = dataset
        del dataset


    def get_dataset(self) -> tf.data.Dataset:
        return self.dataset


    def augmentation(self) -> None:
        if self.config['meta_data']['augmentation_strategy'] == "equal":
            print("Start augmenting")
            num_less_labels = self.train_metadata[self.train_metadata['target'] == 1].shape[0]
            num_more_labels = self.train_metadata[self.train_metadata['target'] == 0].shape[0]

            sample_1_df = self.train_metadata[self.train_metadata['target'] == 1].sample(num_more_labels-num_less_labels, replace=True, random_state=1)

            self.train_metadata = pd.concat([self.train_metadata, sample_1_df], axis=0)

            self.train_metadata = self.train_metadata.sample(frac=1)


    def save_dataset(self) -> None:
        self.dataset.save(self.config['dataset']['save_dataset_path'])


    def load_dataset(self) -> None:
        self.dataset = tf.data.Dataset.load(self.config['dataset']['save_dataset_path'])
