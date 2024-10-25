import pytest
import torch

import rootutils
from src.models.catdog_classifier import (
    ViTTinyClassifier,
)

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from src.models.dogbreed_classifier import DogbreedClassifier


# def test_dogbreed_classifier_forward():
#     model = DogbreedClassifier(model_name="resnet18", pretrained=True, num_classes=10)
#     batch_size, channels, height, width = 4, 3, 224, 224
#     x = torch.randn(batch_size, channels, height, width)
#     output = model(x)
#     assert output.shape == (batch_size, 10)


@pytest.fixture
def model():
    """
    Fixture to initialize the ViTTinyClassifier with default settings.
    Provides a reusable instance across multiple tests.
    """
    return ViTTinyClassifier()


def test_model_initialization():
    # Test initialization with default parameters
    model = ViTTinyClassifier()
    assert model is not None
    assert isinstance(model, ViTTinyClassifier)

    # Test initialization with custom parameters
    custom_model = ViTTinyClassifier(
        img_size=128, num_classes=10, embed_dim=32, depth=4, num_heads=2
    )
    assert custom_model.hparams.img_size == 128
    assert custom_model.hparams.num_classes == 10
    assert custom_model.hparams.embed_dim == 32
    assert custom_model.hparams.depth == 4
    assert custom_model.hparams.num_heads == 2


def test_forward_pass(model):
    # Generate a random input tensor with batch size 4, 3 channels, and 224x224 image size
    x = torch.randn(4, 3, 224, 224)
    output = model(x)

    # Ensure the output is of shape (batch_size, num_classes)
    assert output.shape == (4, model.hparams.num_classes)


def test_training_step(model):
    # Generate a random batch of data
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, model.hparams.num_classes, (4,))
    batch = (x, y)

    # Run the training step and ensure no errors
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_validation_step(model):
    # Generate a random batch of data
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, model.hparams.num_classes, (4,))
    batch = (x, y)

    # Run the validation step and ensure no errors
    model.validation_step(batch, 0)


def test_configure_optimizers(model):
    # Ensure that configure_optimizers returns an optimizer and a learning rate scheduler
    optimizers = model.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers
    assert isinstance(optimizers["optimizer"], torch.optim.Optimizer)
    assert "scheduler" in optimizers["lr_scheduler"]
