
import tensorflow
import hydra
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf

from src.models.tf.simple_model import SimpleModel


def train(config: DictConfig) -> None:
    pass


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    train(config)


if __name__=="__main__":
    main()



