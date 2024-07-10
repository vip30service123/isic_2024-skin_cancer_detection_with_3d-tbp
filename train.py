import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import hydra
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf

from src.data_processing.tf.dataset import Dataset
from src.models.tf.simple_model import SimpleModel


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    do_create_new_dataset = config['do_create_new_dataset']
    dataset = Dataset(config=config)
    if do_create_new_dataset:
        dataset.augmentation()
        dataset.prepare_dataset()
        dataset.save_dataset()
    else:
        dataset.load_dataset()
    print("#### Train test split")
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

    epochs=config['training']['epochs']
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    save_model_path = config['model']['save_model_path']
    model.save(os.path.join(save_model_path, "first.keras"))


if __name__=="__main__":
    main()



