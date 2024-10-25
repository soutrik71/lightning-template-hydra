import sys
from pathlib import Path
import rootutils

# Set up the project root
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Ensure the src directory is in the PYTHONPATH
sys.path.insert(0, str(root / "src"))

import pytest
from src.datamodules.catdog_datamodule import CatDogImageDataModule


@pytest.fixture
def data_module():
    """
    Fixture to initialize the CatDogImageDataModule with test-specific settings.
    Provides a reusable instance across multiple tests.
    """
    return CatDogImageDataModule(
        data_dir="test_data",  # Adjust to your test data path
        num_workers=1,
        batch_size=4,
        splits=(0.7, 0.3),  # Custom split for testing
        pin_memory=False,
        image_size=128,  # Smaller image size for faster testing
    )
