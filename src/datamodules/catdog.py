import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from pathlib import Path
from typing import Union
import os


class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download images and prepare images datasets."""
        download_and_extract_archive(
            url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
            download_root=self.data_dir,
            remove_finished=True,
        )

    def setup(self, stage: str):
        data_path = self.data_dir / "cats_and_dogs_filtered"

        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                root=data_path / "train", transform=self.train_transform
            )
            self.val_dataset = ImageFolder(
                root=data_path / "validation", transform=self.val_transform
            )

        if stage == "test" or stage is None:
            # For this example, we'll use the validation set as the test set
            # In a real scenario, you'd have a separate test set
            self.test_dataset = ImageFolder(
                root=data_path / "validation", transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )
