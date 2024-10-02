import pytest
import os

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datamodules.catdog_datamodule import CatDogImageDataModule


@pytest.fixture
def datamodule():
    return CatDogImageDataModule(data_dir="data", batch_size=8)


def test_catdog_datamodule_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Update this assertion to check for the correct total size
    total_size = len(datamodule.train_dataset) + len(datamodule.val_dataset)
    assert total_size == len(
        datamodule.create_dataset(
            datamodule.data_path / "cats_and_dogs_filtered" / "train",
            datamodule.train_transform,
        )
    )


def test_catdog_datamodule_train_val_test_splits(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    # Check if the splits are correct (80% train, 20% val)
    total_train_val = len(datamodule.train_dataset) + len(datamodule.val_dataset)
    assert len(datamodule.train_dataset) / total_train_val == pytest.approx(
        0.8, abs=0.01
    )
    assert len(datamodule.val_dataset) / total_train_val == pytest.approx(0.2, abs=0.01)

    # Check if test dataset is separate
    assert len(datamodule.test_dataset) > 0


def test_catdog_datamodule_dataloaders(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check if the batch sizes are correct
    assert train_loader.batch_size == 8
    assert val_loader.batch_size == 8
    assert test_loader.batch_size == 8


def test_catdog_datamodule_transforms(datamodule):
    assert datamodule.train_transform is not None
    assert datamodule.valid_transform is not None

    # Check if the image size in transforms matches the specified size
    assert datamodule.train_transform.transforms[0].size == (224, 224)
    assert datamodule.valid_transform.transforms[0].size == (224, 224)


def test_catdog_datamodule_data_path(datamodule):
    assert datamodule.data_path.exists()
    assert (datamodule.data_path / "cats_and_dogs_filtered").exists()
    assert (datamodule.data_path / "cats_and_dogs_filtered" / "train").exists()
    assert (datamodule.data_path / "cats_and_dogs_filtered" / "validation").exists()
