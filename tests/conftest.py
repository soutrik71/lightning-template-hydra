import sys
import os
import pytest
from omegaconf import OmegaConf
import hydra
import rootutils

# Setup root directory to project root and add src/ to the Python path
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datamodules.dogbreed_datamodule import DogbreedDataModule, input_dataprep


@pytest.fixture(scope="session")
def config():
    """Load the configuration from the test.yaml file, once per test session."""
    # Set the absolute path to the configuration file
    config_path = os.path.abspath(os.path.join(root, "configs"))

    # Initialize Hydra and compose configurations from test.yaml and its defaults
    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name="test")

    # Debugging: Print the merged config to verify
    print(OmegaConf.to_yaml(cfg))

    # Ensure that the model key exists
    if "model" not in cfg:
        raise KeyError("Missing key 'model' in config file")

    return cfg


@pytest.fixture(scope="session")
def dataset_df(config):
    """Prepare the dataset once for all tests in the session."""
    return input_dataprep(config)


@pytest.fixture
def datamodule(config, dataset_df):
    """Initialize the DogbreedDataModule for each test."""
    return DogbreedDataModule(
        cfg=config, dataset_df=dataset_df  # Pass the config object as a parameter
    )
