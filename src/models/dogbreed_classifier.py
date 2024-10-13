import timm
import torch.nn.functional as F
from torch import optim
import lightning as L
from torchmetrics import Accuracy
from loguru import logger


class DogbreedClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet18",  # Default model
        lr: float = 0.001,  # Default learning rate
        weight_decay: float = 1e-5,  # Default weight decay
        scheduler_factor: float = 0.1,  # Default scheduler factor
        scheduler_patience: int = 10,  # Default scheduler patience
        min_lr: float = 1e-6,  # Default minimum learning rate
        pretrained: bool = True,  # Whether to use pretrained model
    ):
        """
        Initialize the DogBreedClassifier.

        Args:
            num_classes (int): The number of dog breed classes.
            model_name (str): Name of the pre-trained model (default: "resnet18").
            lr (float): Learning rate for the optimizer (default: 0.001).
            weight_decay (float): Weight decay for the optimizer (default: 1e-5).
            scheduler_factor (float): Factor for learning rate scheduler (default: 0.1).
            scheduler_patience (int): Number of epochs with no improvement after which learning rate will be reduced (default: 10).
            min_lr (float): Minimum learning rate (default: 1e-6).
            pretrained (bool): Whether to use a pre-trained model (default: True).
        """
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr

        # Load the model with dynamic options for model name and pre-trained weights
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
        )

        # Initialize accuracy metrics for multi-class classification
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

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
        self.test_acc(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
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
