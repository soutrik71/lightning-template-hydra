import pytest
import torch

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.dogbreed_classifier import DogbreedClassifier


def test_dogbreed_classifier_forward():
    model = DogbreedClassifier(model_name="resnet18", pretrained=True, num_classes=10)
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, 10)
