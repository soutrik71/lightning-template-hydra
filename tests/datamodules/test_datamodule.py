import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from src.datamodules.catdog_datamodule import CatDogImageDataModule


@patch("src.datamodules.catdog_datamodule.download_and_extract_archive")
def test_prepare_data(mock_download, data_module):
    # Mock the download method to prevent actual downloading
    mock_download.return_value = None

    # Run prepare_data and ensure the download function is called if needed
    data_module.prepare_data()
    mock_download.assert_called_once_with(
        url=data_module.url,
        download_root=data_module.data_path,
        remove_finished=True,
    )


def test_transforms(data_module):
    # Check if the transforms are correctly set up
    train_transform = data_module.train_transform
    valid_transform = data_module.valid_transform

    assert train_transform is not None
    assert valid_transform is not None


def test_dataset_creation(data_module):
    # Mock the dataset creation to avoid needing actual files
    with patch.object(data_module, "create_dataset") as mock_create:
        mock_create.return_value = MagicMock()
        data_module.setup()

        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None


def test_dataloader(data_module):
    with patch.object(data_module, "create_dataset") as mock_create:

        mock_dataset = MagicMock()

        # Ensure the mock dataset returns a non-zero length
        mock_dataset.__len__.return_value = 10  # Simulate 10 samples in the dataset
        mock_create.return_value = mock_dataset

        data_module.setup()

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        # Check that DataLoader objects are returned
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Verify correct batch size
        assert train_loader.batch_size == data_module._batch_size
        assert val_loader.batch_size == data_module._batch_size
        assert test_loader.batch_size == data_module._batch_size

        # Verify DataLoader contains the expected number of batches
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


def test_data_splitting(data_module):
    # Check correct splitting of train and validation data
    with patch.object(data_module, "create_dataset") as mock_create:
        mock_dataset = MagicMock()
        mock_create.return_value = mock_dataset
        mock_dataset.__len__.return_value = 100  # Assume 100 total samples

        data_module.setup()

        # Check if the split is as expected (70% train, 30% val)
        assert len(data_module.train_dataset) == 70
        assert len(data_module.val_dataset) == 30
