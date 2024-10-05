import sys
import os
import pytest
from omegaconf import OmegaConf

import rootutils

# Setup root directory to project root and add src/ to the Python path
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datamodules.dogbreed_datamodule import DogbreedDataModule, input_dataprep


@pytest.fixture(scope="session")
def config():
    """Load the configuration from the train.yaml file, once per test session."""
    # Set the absolute path to the configuration file
    config_path = os.path.abspath(os.path.join(root, "configs/test.yaml"))

    # Load configuration using OmegaConf
    config = OmegaConf.load(config_path)

    # Debugging: Print the config to verify
    print(OmegaConf.to_yaml(config))

    # Ensure that the paths key exists
    if "defaults" not in config:
        raise KeyError("Missing key 'paths' in config file")

    return config


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
