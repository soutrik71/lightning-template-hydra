import pytest
from unittest.mock import patch, MagicMock
from src.train import (
    instantiate_callbacks,
    instantiate_loggers,
    load_checkpoint_if_available,
    clear_checkpoint_directory,
    train_module,
    run_test_module,
)
from pathlib import Path
import lightning as L
from omegaconf import OmegaConf, DictConfig


@pytest.fixture
def dummy_cfg():
    """Create a dummy configuration for testing."""
    return DictConfig(
        {
            "paths": {
                "ckpt_dir": "test_checkpoints",
                "log_dir": "test_logs",
                "artifact_dir": "test_artifacts",
            },
            "ckpt_path": None,  # Add this line to fix the missing key issue
            "callbacks": {},
            "logger": {},
            "model": {"_target_": "src.models.catdog_classifier.ViTTinyClassifier"},
            "trainer": {"_target_": "lightning.Trainer", "max_epochs": 1},
            "train": True,
            "test": True,
            "task_name": "train",
            "seed": 42,
        }
    )


@patch("src.train.hydra.utils.instantiate")
def test_instantiate_callbacks(mock_instantiate, dummy_cfg):
    mock_instantiate.return_value = MagicMock()
    callbacks = instantiate_callbacks(dummy_cfg.callbacks)
    assert isinstance(callbacks, list)
    assert len(callbacks) == 0  # Assuming empty callbacks config for this test


@patch("src.train.hydra.utils.instantiate")
def test_instantiate_loggers(mock_instantiate, dummy_cfg):
    mock_instantiate.return_value = MagicMock()
    loggers = instantiate_loggers(dummy_cfg.logger)
    assert isinstance(loggers, list)
    assert len(loggers) == 0  # Assuming empty logger config for this test


def test_load_checkpoint_if_available():
    path = "test_checkpoint.ckpt"
    Path(path).touch()  # Create an empty file
    assert load_checkpoint_if_available(path) == path
    Path(path).unlink()  # Remove the file
    assert load_checkpoint_if_available(path) is None


def test_clear_checkpoint_directory(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "file1.ckpt").touch()
    (ckpt_dir / "file2.ckpt").touch()

    clear_checkpoint_directory(str(ckpt_dir))
    assert len(list(ckpt_dir.iterdir())) == 0


@patch("src.train.logger.info")
def test_train_module(mock_logger, dummy_cfg):
    # Mock model, datamodule, and trainer
    model = MagicMock(spec=L.LightningModule)
    datamodule = MagicMock(spec=L.LightningDataModule)

    # Create an actual instance of Trainer with `fit` as a real method
    with patch("src.train.L.Trainer.fit") as mock_fit:
        trainer = L.Trainer()

        # Run the training module
        train_module(dummy_cfg, datamodule, model, trainer)

        # Assert that the `fit` method was called once
        mock_fit.assert_called_once()


@patch("src.train.logger.info")
def test_run_test_module(mock_logger, dummy_cfg):
    # Mock model, datamodule, and trainer
    model = MagicMock(spec=L.LightningModule)
    datamodule = MagicMock(spec=L.LightningDataModule)

    # Create an actual instance of Trainer with `test` as a real method
    with patch.object(
        L.Trainer, "test", return_value=[{"test_loss": 0.5, "test_acc": 0.8}]
    ) as mock_test:
        trainer = L.Trainer()

        # Run the testing module
        run_test_module(dummy_cfg, datamodule, model, trainer)

        # Assert that the `test` method was called once
        mock_test.assert_called_once()
