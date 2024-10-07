import pytest
from unittest.mock import MagicMock
import torch
import lightning as L
from src.train import (
    instantiate_callbacks,
    instantiate_loggers,
    train_module,
    test,
    setup_run_trainer,
)


# Mocking Hydra's utils and trainer for the tests
@pytest.fixture
def mock_hydra_utils(mocker):
    # Mock hydra.utils.instantiate to accept any arguments and return a mock
    mocker.patch(
        "hydra.utils.instantiate", side_effect=lambda *args, **kwargs: MagicMock()
    )
    return mocker


@pytest.fixture
def mock_trainer():
    """Mock the trainer to bypass actual model training."""
    trainer = MagicMock(spec=L.Trainer)
    trainer.callback_metrics = {"loss": 0.1, "accuracy": 0.9}
    trainer.checkpoint_callback.best_model_path = "best_checkpoint.ckpt"
    return trainer


@pytest.fixture
def mock_datamodule():
    """Mock the DataModule for training and testing."""
    datamodule = MagicMock(spec=L.LightningDataModule)
    return datamodule


@pytest.fixture
def mock_model():
    """Mock the model."""
    model = MagicMock(spec=L.LightningModule)
    return model


# Test instantiate_callbacks function
def test_instantiate_callbacks(mock_hydra_utils, config):
    callbacks = instantiate_callbacks(config.get("callbacks"))
    assert isinstance(callbacks, list)
    # Check that the number of callbacks matches the configuration (4 callbacks in this case)
    assert len(callbacks) == 4  # Expecting 4 callbacks based on the config


# Test instantiate_loggers function
def test_instantiate_loggers(mock_hydra_utils, config):
    loggers = instantiate_loggers(config.get("logger"))
    assert isinstance(loggers, list)
    assert len(loggers) == 0  # No loggers in mock config


# Test train_module function
def test_train_module(mock_trainer, mock_model, mock_datamodule, config):
    train_module(config, mock_datamodule, mock_model, mock_trainer)
    mock_trainer.fit.assert_called_once_with(mock_model, mock_datamodule)
    assert "loss" in mock_trainer.callback_metrics
    assert "accuracy" in mock_trainer.callback_metrics


# Test test function
def test_test_module(mock_trainer, mock_model, mock_datamodule, config):
    test(config, mock_datamodule, mock_model, mock_trainer)
    mock_trainer.test.assert_called_with(
        mock_model, mock_datamodule, ckpt_path="best_checkpoint.ckpt"
    )


# Integration test for setup_run_trainer using test.yaml configuration
def test_setup_run_trainer(
    mock_hydra_utils, config, mock_trainer, mock_datamodule, mock_model, mocker
):
    # Ensure the model config exists
    assert "model" in config, "Model configuration is missing"

    # Mock the dataloader and other utilities
    mocker.patch(
        "src.train.main_dataloader", return_value=(MagicMock(), mock_datamodule)
    )
    mocker.patch("src.utils.logging_utils.setup_logger")
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("lightning.seed_everything")

    # Run the trainer setup with the test.yaml configuration
    setup_run_trainer(config)

    # Assertions to verify correct behavior
    # assert torch.cuda.is_available() == False
    L.seed_everything.assert_called_once_with(config.seed, workers=True)
