import pytest
from hydra import initialize, compose
from src.train import train_and_test
import rootutils


def test_train_and_test():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=catdog_ex",
                "+trainer.fast_dev_run=True",
                "test=True",
            ],
        )

    # Override paths in the config
    project_root = rootutils.find_root(indicator=".project-root")
    cfg.paths.root_dir = str(project_root)
    cfg.paths.data_dir = str(project_root / "data")
    cfg.paths.log_dir = str(project_root / "logs")
    cfg.paths.output_dir = str(project_root / "outputs")
    cfg.paths.work_dir = str(project_root)

    # Run the training function
    metrics = train_and_test(cfg)

    # Check if metrics are returned
    assert isinstance(metrics, dict), "train_and_test should return a dictionary"
    assert "train" in metrics, "Metrics should contain 'train' key"
    assert "test" in metrics, "Metrics should contain 'test' key"

    # Check if train metrics exist
    assert len(metrics["train"]) > 0, "Train metrics should not be empty"

    # Check if specific metrics exist in train metrics
    assert "train/loss" in metrics["train"], "Train loss should be in metrics"
    assert "train/acc" in metrics["train"], "Train accuracy should be in metrics"

    # Check if test metrics exist (since we set test=True)
    assert len(metrics["test"]) > 0, "Test metrics should not be empty"

    # Check if specific metrics exist in test metrics
    assert "test/loss" in metrics["test"], "Test loss should be in metrics"
    assert "test/acc" in metrics["test"], "Test accuracy should be in metrics"

    # Optional: Check if metric values are within expected ranges
    assert (
        0 <= metrics["train"]["train/loss"] <= 10
    ), "Train loss should be between 0 and 10"
    assert (
        0 <= metrics["train"]["train/acc"] <= 1
    ), "Train accuracy should be between 0 and 1"
    assert (
        0 <= metrics["test"]["test/loss"] <= 10
    ), "Test loss should be between 0 and 10"
    assert (
        0 <= metrics["test"]["test/acc"] <= 1
    ), "Test accuracy should be between 0 and 1"
