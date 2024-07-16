"""
Single Responsibility Principal:
- Most of methods in this file are independent to the rest.
- Split every method into sub-class.
Open/Close Pricipal:
- My Augmentation class will not satisfy open/close principal because there will be a lot of if-else for each type of augmentaion method.
- Make a checking method, and use Class.__subclasses__() to iterate and find suitable method.
- If I want to add a new augmentation class, just extent the checking method and I am done.
Liskov's substitution principle:
- State that there is a series of properties that an object type must hold to preserve reliability on its design.
- If S is exteneded from T, then object of type T can be replace with object of type S without breaking any behaviors.
Interface segregation:

"""

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
    """
    Based class for data processing.
    """



class SuffleDataset(DatasetProcessor):
    def __call__(self, df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
        assert type(df) == pd.DataFrame, f"df must be of type pd.DataFrame, not {type(df)}"
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        return df.sample(frac=1, random_state=config['dataset']['seed'])



class GetMainAttribute(DatasetProcessor):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        assert type(df) == pd.DataFrame, f"df must be of type pd.DataFrame, not {type(df)}"

        return df[['isic_id', 'target']]



class Augmentation(DatasetProcessor):
    @staticmethod
    def check_event(event_type: str) -> bool:
        assert type(event_type) == str, f"event_type must be of type string, not {type(event_type)}"

        return False


    def __call__(self, df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
        assert type(df) == pd.DataFrame, f"df must be of type pd.DataFrame, not {type(df)}"
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        pass



class EqualAugmentation(Augmentation):
    @staticmethod
    def check_event(event_type: str) -> bool:
        assert type(event_type) == str, f"df must be of type string, not {type(event_type)}"

        return event_type == "equal"


    def __call__(self, df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
        assert type(df) == pd.DataFrame, f"df must be of type pd.DataFrame, not {type(df)}"
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        num_less_labels = df[df['target'] == 1].shape[0]
        num_more_labels = df[df['target'] == 0].shape[0]

        less_labels_df = df[df['target'] == 1].copy()
        less_labels_df = less_labels_df.sample(num_more_labels-num_less_labels, replace=True, random_state=config['dataset']['seed'])

        return pd.concat([df, less_labels_df], axis=0)



class Augmentating(DatasetProcessor):
    def __init__(self, augmentation_type: str = 'equal'):
        assert type(augmentation_type) == str, f"augmentation_type must be of type string, not {type(augmentation_type)}"

        self.augmentation_type = augmentation_type


    def __call__(self, df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
        assert type(df) == pd.DataFrame, f"df must be of type pd.DataFrame, not {type(df)}"
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        for augmentation_class in Augmentation.__subclasses__():
            if augmentation_class.check_event(self.augmentation_type):
                return augmentation_class()(
                    df=df,
                    config=config
                )



class TrainTestSplit(DatasetProcessor):
    def __call__(self, df: pd.DataFrame, config: DictConfig) -> Tuple:
        assert type(df) == pd.DataFrame, f"df must be of type pd.DataFrame, not {type(df)}"
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        train_rs, _ = config['dataset']['train_test_split']

        df_length = df.shape[0]

        train_sz = int(df_length * train_rs)

        train_df = df.iloc[:train_sz]
        test_df = df.iloc[train_sz:]

        return train_df, test_df



class GetTrainMetadataDF(DatasetProcessor):
    def __call__(self, config: DictConfig) -> pd.DataFrame:
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        return pd.read_csv(config['meta_data']['train_meta_data_path'])



class GetTestMetadataDF(DatasetProcessor):
    def __call__(self, config: DictConfig) -> pd.DataFrame:
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        return pd.read_csv(config['meta_data']['test_meta_data_path'])
