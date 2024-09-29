import os
import yaml
from pydantic.v1 import BaseModel, Field, BaseSettings
from loguru import logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))


class DataConfig(BaseModel):
    data_dir: str = Field(..., description="Path to the data directory")
    kaggle_dataset_path: str = Field(..., description="Path to the kaggle dataset")
    artifact_path: str = Field(..., description="Path to the artifact directory")


class DataLoaderConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for the DataLoader")
    num_workers: int = Field(..., description="Number of workers for the DataLoader")


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    lr: float = Field(..., description="Learning rate for the optimizer")
    weight_decay: float = Field(..., description="Weight decay for the optimizer")
    scheduler_factor: float = Field(
        ..., description="Scheduler factor for the optimizer"
    )
    scheduler_patience: int = Field(
        ..., description="Scheduler patience for the optimizer"
    )
    min_lr: float = Field(..., description="Minimum learning rate for the optimizer")


class TrainConfig(BaseModel):
    seed: int = Field(..., description="Seed for reproducibility")
    num_epochs: int = Field(..., description="Maximum number of epochs")
    checkpoint_path: str = Field(..., description="Path to the checkpoint directory")


class Settings(BaseSettings):
    kaggle_username: str = os.environ.get("KAGGLE_USERNAME")
    kaggle_key: str = os.environ.get("KAGGLE_KEY")
    data_config: DataConfig
    dataloader_config: DataLoaderConfig
    model_config: ModelConfig
    train_config: TrainConfig


def load_yaml_config() -> dict:
    # Load static values from config.yaml
    config_path = f"src/config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from service path, trying config.yaml: {e}")
        try:
            with open("config.yaml", "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading fallback config.yaml: {e}")
            raise


yaml_config = load_yaml_config()
settings = Settings(**yaml_config)

if __name__ == "__main__":
    print(settings)
