import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import hydra
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf

from src.data_processing.tf.dataset import Dataset
from src.models.tf.simple_model import SimpleModel


def train(config: DictConfig) -> None:
    pass


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    dataset = Dataset(config=config)
    dataset.prepare_dataset()
    train_ds, val_ds, test_ds = dataset.train_test_split()
    print(len(list(train_ds)), len(list(val_ds)), len(list(test_ds)))

    model = tf.keras.applications.ResNet50(
        include_top=config['model']['include_top'],
        weights=config['model']['weights'],
        input_tensor=config['model']['input_tensor'],
        input_shape=config['model']['input_shape'],
        pooling=config['model']['pooling'],
        classes=config['model']['classes'],
        classifier_activation=config['model']['classifier_activation']
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

    epochs=10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


if __name__=="__main__":
    main()



