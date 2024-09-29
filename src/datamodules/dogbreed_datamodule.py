import os
import lightning as L
from torchvision import transforms
from pathlib import Path
from pydantic.v1 import BaseModel, Field, BaseSettings
import zipfile
import yaml, shutil
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from src.settings import settings
from pydantic.v1 import BaseSettings
from loguru import logger
import kaggle


def download_data_kaggle(settings: BaseSettings):
    """Download the dataset from Kaggle."""
    # Download the dataset if the file does not exist
    if not os.path.exists(
        f'{settings.data_config.kaggle_dataset_path.split("/")[-1]}.zip'
    ):
        logger.info("Downloading the dataset from Kaggle")
        kaggle.api.dataset_download_files(
            settings.data_config.kaggle_dataset_path, path="./", unzip=False
        )
    if not os.path.exists(settings.data_config.data_dir):
        logger.info("Data directoy doesnot exists")
        os.makedirs(settings.data_config.data_dir)

    # Unzip the dataset
    with zipfile.ZipFile(
        f'{settings.data_config.kaggle_dataset_path.split("/")[-1]}.zip', "r"
    ) as zip_ref:
        zip_ref.extractall(
            os.path.join(
                settings.data_config.data_dir,
                settings.data_config.kaggle_dataset_path.split("/")[-1],
            )
        )


def input_dataprep(settings: BaseSettings):
    # Download the dataset
    download_data_kaggle(settings)
    # Path to the training dataset

    TRAIN_PATH = Path(
        f"{settings.data_config.data_dir}/{settings.data_config.kaggle_dataset_path.split('/')[-1]}/dataset"
    )
    IMAGE_PATH_LIST = list(TRAIN_PATH.glob("*/*.jpg"))
    logger.info(f"Total Images = {len(IMAGE_PATH_LIST)}")

    # creating a mapping file for the classes
    images_path = [None] * len(IMAGE_PATH_LIST)
    labels = [None] * len(IMAGE_PATH_LIST)

    for i, img_path in enumerate(IMAGE_PATH_LIST):
        images_path[i] = img_path
        labels[i] = img_path.parent.stem

    dataset_df = pd.DataFrame({"image_path": images_path, "label": labels})

    return dataset_df


class DogbreedDataset(Dataset):
    """Dog breed dataset."""

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transform: A function/transform that takes in a PIL image and returns a transformed version.
        """
        super().__init__()

        self.paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

        self.classes = sorted(df["label"].unique())
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }  # Map class names to indices

    def load_image(self, index: int) -> Image.Image:
        """Load image from file."""
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.paths)

    def __getitem__(self, index: int):
        """Retrieve an image and its corresponding label."""
        image = self.load_image(index)
        class_name = self.labels[index]
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, class_idx


class DogbreedDataModule(L.LightningDataModule):
    """DataModule for the Dog Breed Classifier."""

    def __init__(
        self,
        dataset_df,
        num_workers: int = settings.dataloader_config.num_workers,
        batch_size: int = settings.dataloader_config.batch_size,
    ):
        """
        Initialize the DataModule with dataset DataFrame, batch size, and number of workers.

        Args:
            dataset_df (pd.DataFrame): DataFrame containing image paths and labels.
            num_workers (int): Number of workers for data loading.
            batch_size (int): Batch size for DataLoader.
        """
        super().__init__()
        self.dataset_df = dataset_df  # Store the DataFrame with image paths and labels
        self.num_workers = num_workers  # Set number of workers for data loading
        self.batch_size = batch_size  # Set the batch size

    @property
    def normalize_transform(self):
        """Return normalization transform for the images."""
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        """Return a composition of data augmentations and transformations for training."""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize image to 224x224
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.ToTensor(),  # Convert to tensor
                self.normalize_transform,  # Normalize the image
            ]
        )

    def setup(self, stage: str = None):
        """
        Called by Lightning to initialize the datasets for the train, val, and test stages.
        This method sets up the datasets depending on the current stage.

        Args:
            stage (str): Either 'fit', 'test', or None.
        """
        dataset = DogbreedDataset(self.dataset_df, transform=self.train_transform)

        if stage == "fit" or stage is None:
            # Split dataset into train (80%) and val (10%)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size]
            )

        if stage == "test":
            # use the val_dataset for testing
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size]
            )

    def train_dataloader(self):
        """Return the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle data during training
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Return the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Do not shuffle validation data
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Return the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Do not shuffle test data
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dataset_df = input_dataprep(settings)
    dogbreed_datamodule = DogbreedDataModule(dataset_df)
    dogbreed_datamodule.setup()
    train_loader = dogbreed_datamodule.train_dataloader()
    val_loader = dogbreed_datamodule.val_dataloader()
    logger.info("DataModule setup successful")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
