import os
import lightning as L
from torchvision import transforms
from pathlib import Path
from pydantic.v1 import BaseSettings
import zipfile
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from loguru import logger
import kaggle
import hydra
from omegaconf import DictConfig


def download_data_kaggle(settings: BaseSettings):
    """Download the dataset from Kaggle."""
    if not os.path.exists(
        f'{settings.data_config.kaggle_dataset_path.split("/")[-1]}.zip'
    ):
        logger.info("Downloading the dataset from Kaggle")
        kaggle.api.dataset_download_files(
            settings.data_config.kaggle_dataset_path, path="./", unzip=False
        )
    if not os.path.exists(settings.data_config.data_dir):
        logger.info("Data directory does not exist, creating it.")
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
    """Prepare the input data."""
    download_data_kaggle(settings)
    # Path to the dataset
    dataset_path = Path(
        f"{settings.data_config.data_dir}/{settings.data_config.kaggle_dataset_path.split('/')[-1]}/dataset"
    )
    image_path_list = list(dataset_path.glob("*/*.jpg"))
    logger.info(f"Total Images = {len(image_path_list)}")

    images_path = [str(img_path) for img_path in image_path_list]
    labels = [img_path.parent.stem for img_path in image_path_list]

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
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

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
        dataset_df: pd.DataFrame,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        img_size: int = 224,
        num_workers: int = 4,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Args:
            dataset_df (pd.DataFrame): DataFrame containing image paths and labels.
            train_split (float): Percentage of data to use for training.
            val_split (float): Percentage of data to use for validation.
            test_split (float): Percentage of data to use for testing.
            img_size (int): Size to resize images to.
            num_workers (int): Number of workers for data loading.
            batch_size (int): Batch size for DataLoader.
        """
        super().__init__()
        assert train_split + val_split + test_split == 1.0, "Splits must sum to 1.0"
        self.dataset_df = dataset_df
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.img_size = img_size
        self.num_workers = num_workers
        self.batch_size = batch_size

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
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def setup(self, stage: str = None):
        """
        Initialize datasets for the train, val, and test stages.
        Args:
            stage (str): Either 'fit', 'test', or None.
        """
        dataset = DogbreedDataset(self.dataset_df, transform=self.train_transform)

        # Split dataset
        total_size = len(dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        """Return the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Return the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Return the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dataset_df = input_dataprep()
    dogbreed_datamodule = DogbreedDataModule(
        dataset_df, train_split=0.7, val_split=0.2, test_split=0.1, img_size=224
    )
    dogbreed_datamodule.setup()
    train_loader = dogbreed_datamodule.train_dataloader()
    val_loader = dogbreed_datamodule.val_dataloader()
    logger.info("DataModule setup successful")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
