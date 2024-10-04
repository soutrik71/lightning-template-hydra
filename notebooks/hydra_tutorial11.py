import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str = "fashion_mnist"
    batch_size: int = 128


@dataclass
class ModelConfig:
    architecture: str = "rcnn"
    layers: int = 10


@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()


@hydra.main(version_base=None, config_path=None, config_name="config")
def run_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run_model()
