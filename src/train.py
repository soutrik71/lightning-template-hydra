import os
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
from src.datamodules.dogbreed_datamodule import main_dataloader
from src.utils.logging_utils import setup_logger, task_wrapper
from loguru import logger
from dotenv import load_dotenv, find_dotenv
import rootutils
import hydra
from omegaconf import DictConfig


# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    """Instantiate and return a list of callbacks from the configuration."""
    callbacks: List[L.Callback] = []

    if not callback_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiate and return a list of loggers from the configuration."""
    loggers_ls: List[Logger] = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping..")
        return loggers_ls

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers_ls.append(hydra.utils.instantiate(lg_conf))

    return loggers_ls


@task_wrapper
def train_module(
    cfg: DictConfig,
    data_module: L.LightningDataModule,
    model: L.LightningModule,
    trainer: L.Trainer,
):
    """Train the model using the provided Trainer and DataModule."""
    logger.info("Training the model")
    trainer.fit(model, data_module)
    train_metrics = trainer.callback_metrics
    logger.info(f"Training metrics:\n{train_metrics}")


@task_wrapper
def test(
    cfg: DictConfig,
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    trainer: L.Trainer,
):
    """Test the model using the best checkpoint or the current model weights."""
    logger.info("Testing the model")
    datamodule.setup(stage="test")

    if trainer.checkpoint_callback.best_model_path:
        logger.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        logger.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)

    logger.info(f"Test metrics:\n{test_metrics}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def setup_run_trainer(cfg: DictConfig):
    """Set up and run the Trainer for training and testing the model."""
    # Initialize logger
    log_path = Path(cfg.paths.log_dir) / "train.log"
    setup_logger(log_path)

    # Initialize DataModule
    logger.info("Setting up the DataModule")
    dataset_df, dogbreed_datamodule = main_dataloader(cfg)
    labels = dataset_df.label.nunique()
    logger.info(f"Number of classes: {labels}")

    os.makedirs(cfg.paths.artifact_dir, exist_ok=True)
    dataset_df.to_csv(
        Path(cfg.paths.artifact_dir) / "dogbreed_dataset.csv", index=False
    )

    # Check for GPU availability
    logger.info("GPU available" if torch.cuda.is_available() else "No GPU available")

    # Set seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)

    # Initialize model
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info(f"Model summary:\n{model}")

    # Set up callbacks and loggers
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    # Train and test the model based on config settings
    if cfg.get("train"):
        train_module(cfg, dogbreed_datamodule, model, trainer)

    if cfg.get("test"):
        test(cfg, dogbreed_datamodule, model, trainer)

    # Write training done flag using Hydra paths config
    done_flag_path = Path(cfg.paths.ckpt_dir) / "train_done.flag"
    with done_flag_path.open("w") as f:
        f.write("Training completed.\n")
    logger.info(f"Training completion flag written to: {done_flag_path}")


if __name__ == "__main__":
    setup_run_trainer()
