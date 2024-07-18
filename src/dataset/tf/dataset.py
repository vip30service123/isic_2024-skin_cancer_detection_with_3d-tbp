"""
Outdated,just put here, not use
"""


import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
from typing import Any, List

import h5py
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import tensorflow as tf



class DatasetFromGenerator:
    def __call__(
        self, 
        isic_ids: List[str], 
        labels: List[int], 
        config: DictConfig
    ) -> tf.data.Dataset:
        """Create generator and load with tf.data.Dataset.
        
        Args:
            isic_ids: isic ids to load image from hdf5 file.
            labels: isic item label.
            config: config.
        
        Returns:
            Return tf.data.Dataset that load from generators 
        """

        assert type(isic_ids) == list, f" isic_ids must be of type list, not {type(isic_ids)}"
        assert type(labels) == list, f"labels must be of type list, not {type(labels)}"
        assert type(config) == DictConfig, f"config must be of type Dictconfig, not {type(config)}"

        if len(isic_ids) != len(labels):
            raise Exception("isic_ids and labels must have the same amount of item.")

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

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(h, w, c), dtype=tf.uint8),
                tf.TensorSpec(shape=(2), dtype=tf.int32))
        )