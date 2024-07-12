import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import io
from tqdm.auto import tqdm
from typing import Any, List, Tuple

import h5py
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


class DatasetProcessor:
    @staticmethod
    def shuffle_df(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
        return df.sample(frac=1, random_state=config['dataset']['seed'])


    @staticmethod
    def get_id_target_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[['isic_id', 'target']]


    @staticmethod
    def augmentation(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
        num_less_labels = df[df['target'] == 1].shape[0]
        num_more_labels = df[df['target'] == 0].shape[0]

        less_labels_df = df[df['target'] == 1].copy()
        less_labels_df = less_labels_df.sample(num_more_labels-num_less_labels, replace=True, random_state=config['dataset']['seed'])

        return pd.concat([df, less_labels_df], axis=0)


    @staticmethod
    def train_test_split(df: pd.DataFrame, config: DictConfig) -> Tuple:
        train_rs, _ = config['dataset']['train_test_split']

        df_length = df.shape[0]

        train_sz = int(df_length * train_rs)

        train_df = df.iloc[:train_sz]
        test_df = df.iloc[train_sz:]

        return train_df, test_df


    @staticmethod
    def dataset_from_generator(isic_ids: List[str], labels: List[int], config: DictConfig):
        def generator() -> Any:
            with h5py.File(config['dataset']['directory'], "r") as f:
                for id, label in zip(isic_ids, labels):
                    im = Image.open(io.BytesIO(f[id][()]))

                    h, w, _ = config['model']['input_shape']
                    im = im.resize((h, w))
                    pix = np.array(im)

                    label = [1, 0] if label == 0 else [0, 1]

                    yield pix, label

        h, w, c = config['model']['input_shape']

        return tf.data.Dataset.from_generator(generator,
                                              output_signature=(
                                                        tf.TensorSpec(shape=(h, w, c), dtype=tf.uint8),
                                                        tf.TensorSpec(shape=(2), dtype=tf.int32))
                                              )


    @staticmethod
    def get_train_metadata_df(config: DictConfig) -> pd.DataFrame:
        return pd.read_csv(config['meta_data']['train_meta_data_path'])


    @staticmethod
    def get_test_metadata_df(config: DictConfig) -> pd.DataFrame:
        return pd.read_csv(config['meta_data']['test_meta_data_path'])
