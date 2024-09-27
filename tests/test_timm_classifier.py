import pytest
import torch
from src.models.timm_classifier import TimmClassifier


def test_timm_classifier():
    model = TimmClassifier(model_name="resnet18", num_classes=2)

    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)

    # Add assertions to check if the output is correct
    assert output.shape == (batch_size, 2)

    # Test training step
    batch = (input_tensor, torch.randint(0, 2, (batch_size,)))
    loss = model.training_step(batch, 0)

    # Add assertions to check if the loss is calculated correctly
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
