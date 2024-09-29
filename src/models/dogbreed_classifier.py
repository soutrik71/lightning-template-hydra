import timm
import torch.nn.functional as F
from torch import nn, optim

import lightning as L

from torchmetrics import Accuracy
from src.settings import settings
from pydantic.v1 import BaseSettings
from loguru import logger


class DogbreedClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = settings.model_config.lr,
        weight_decay: float = settings.model_config.weight_decay,
        scheduler_factor: float = settings.model_config.scheduler_factor,
        scheduler_patience: int = settings.model_config.scheduler_patience,
        min_lr: float = settings.model_config.min_lr,
    ):
        """
        Initialize the DogBreedClassifier.

        Args:
            num_classes (int): The number of dog breed classes.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr

        # Load pre-trained ResNet18 model and adjust the final layer for the correct number of classes
        self.model = timm.create_model(
            settings.model_config.model_name,
            pretrained=True,
            num_classes=self.num_classes,
        )

        # Initialize accuracy metrics for multi-class classification
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Use logits to compute accuracy
        self.train_acc(logits, y)

        # Log loss and accuracy
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step logic."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Use logits to compute accuracy
        self.val_acc(logits, y)

        # Log validation loss and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Use logits to compute accuracy
        self.val_acc(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
        }


if __name__ == "__main__":
    model = DogbreedClassifier(num_classes=10)
    logger.info("Model initialized successfully")
    logger.info(f"Model: {model}")
    logger.info(f"Hyperparameters: {model.hparams}")
    logger.info(f"Learning rate: {model.lr}")
    logger.info(f"Weight decay: {model.weight_decay}")
    logger.info(f"Scheduler factor: {model.scheduler_factor}")
    logger.info(f"Scheduler patience: {model.scheduler_patience}")
    logger.info(f"Minimum learning rate: {model.min_lr}")
    logger.info(f"Number of classes: {model.num_classes}")
    logger.info(f"Model state dict keys: {model.state_dict().keys()}")
