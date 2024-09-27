import pytest
from src.datamodules.catdog_datamodule import CatDogImageDataModule


def test_catdog_datamodule():
    datamodule = CatDogImageDataModule(data_dir="data/catdog_test", batch_size=4)

    # Test prepare_data
    datamodule.prepare_data()

    # Test setup
    datamodule.setup()

    # Test dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Add assertions to check if the dataloaders are correctly set up
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
