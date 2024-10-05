import pytest
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datamodules.dogbreed_datamodule import DogbreedDataModule, main_dataloader


@pytest.mark.usefixtures("datamodule")
def test_dogbreed_datamodule_setup(datamodule):
    """Test that the DogbreedDataModule is correctly set up."""
    datamodule.setup()
    assert len(datamodule.train_dataset) > 0, "Training dataset should not be empty"
    assert len(datamodule.val_dataset) > 0, "Validation dataset should not be empty"
    assert len(datamodule.test_dataset) > 0, "Test dataset should not be empty"


@pytest.mark.usefixtures("datamodule")
def test_dogbreed_datamodule_dataloaders(datamodule):
    """Test that the dataloaders return non-empty batches."""
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert len(train_loader) > 0, "Train DataLoader should return batches"
    assert len(val_loader) > 0, "Validation DataLoader should return batches"
    assert len(test_loader) > 0, "Test DataLoader should return batches"


@pytest.mark.usefixtures("config")
def test_main_dataloader(config):
    """Test that the main_dataloader function returns a valid dataset and datamodule."""
    dataset_df, datamodule = main_dataloader(config)
    assert len(dataset_df) > 0, "The dataset DataFrame should not be empty"
    assert isinstance(
        datamodule, DogbreedDataModule
    ), "The datamodule should be of type DogbreedDataModule"
