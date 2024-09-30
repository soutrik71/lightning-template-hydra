import os
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.datamodules.dogbreed_datamodule import DogbreedDataModule, input_dataprep
from src.models.dogbreed_classifier import DogbreedClassifier
from src.utils.logging_utils import setup_logger, task_wrapper
from src.settings import settings
from loguru import logger

# Set up the logger
setup_logger("logs/train.log")


@task_wrapper
def train_and_test(data_module, model, trainer):
    logger.info("Training the model")
    trainer.fit(model, data_module)
    logger.info("Testing the model")
    data_module.setup(stage="test")
    trainer.test(model, data_module)


@task_wrapper
def setup_run_trainer():
    # check for GPU availability
    if torch.cuda.is_available():
        logger.info("GPU available")
    else:
        logger.info("No GPU available")
    # Set the seed for reproducibility
    L.seed_everything(settings.train_config.seed, workers=True)
    logger.info("Downloading the dataset if required")
    dataset_df = input_dataprep(settings)
    os.makedirs(settings.data_config.artifact_path, exist_ok=True)
    dataset_df.to_csv(
        Path(settings.data_config.artifact_path) / "dogbreed_dataset.csv", index=False
    )
    logger.info("Setting up the DataModule")
    dogbreed_datamodule = DogbreedDataModule(dataset_df)
    dogbreed_datamodule.setup()
    labels = dataset_df.label.nunique()
    logger.info(f"Number of classes: {labels}")
    logger.info("Setting up the model")
    model = DogbreedClassifier(num_classes=labels)

    # Initialize ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Initialize RichProgressBar
    rich_progress_bar = RichProgressBar()

    # Initialize RichModelSummary
    rich_model_summary = RichModelSummary()

    # Initialize Trainer
    logger.info("Setting up the Trainer")
    trainer = L.Trainer(
        max_epochs=settings.train_config.num_epochs,
        callbacks=[checkpoint_callback, rich_progress_bar, rich_model_summary],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=TensorBoardLogger(save_dir="logs", name="dogbreed_classifier"),
        devices="auto",
        log_every_n_steps=settings.train_config.log_every_n_steps,
    )

    train_and_test(dogbreed_datamodule, model, trainer)

    # After training completes:
    with open("./checkpoints/train_done.flag", "w") as f:
        f.write("Training completed.\n")


if __name__ == "__main__":
    setup_run_trainer()
