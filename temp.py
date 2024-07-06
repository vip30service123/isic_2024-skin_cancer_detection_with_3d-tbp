import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

tf.keras.preprocessing.image_dataset_from_directory("data/raw/")