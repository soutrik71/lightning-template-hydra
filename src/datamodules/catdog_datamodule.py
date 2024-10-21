from pathlib import Path
from typing import Union, Tuple
import os

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
        splits: Tuple[float, float] = (0.8, 0.2),  # Changed to two splits
        pin_memory: bool = False,
        image_size: int = 224,
        url: str = "https://download.pytorch.org/tutorials/cats_and_dogs_filtered.zip",
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._splits = splits
        self._pin_memory = pin_memory
        self._dataset = None
        self._image_size = image_size
        self.url = url

    def prepare_data(self):
        """Download images if not already downloaded and extracted."""
        dataset_path = self.data_path / "cats_and_dogs_filtered"
        if not dataset_path.exists():
            download_and_extract_archive(
                url=self.url,
                download_root=self._data_dir,
                remove_finished=True,
            )

    @property
    def data_path(self):
        return self._data_dir

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self._image_size, self._image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self._image_size, self._image_size)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def setup(self, stage: str = None):
        if self._dataset is None:
            train_dataset = self.create_dataset(
                self.data_path / "cats_and_dogs_filtered" / "train",
                self.train_transform,
            )
            train_size = int(self._splits[0] * len(train_dataset))
            val_size = len(train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )

            self.test_dataset = self.create_dataset(
                self.data_path / "cats_and_dogs_filtered" / "validation",
                self.valid_transform,
            )

    def __dataloader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=shuffle,
            pin_memory=self._pin_memory,
        )

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset)


if __name__ == "__main__":
    import os
    from pathlib import Path
    import logging

    import hydra
    from omegaconf import DictConfig, OmegaConf
    import lightning as L

    import rootutils

    # Setup root directory
    root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    log = logging.getLogger(__name__)

    print(f"Root directory: {root}")

    @hydra.main(
        version_base="1.3",
        config_path=str(root / "configs"),
        config_name="train",
    )
    def main(cfg: DictConfig):
        # print the config
        log.info(OmegaConf.to_yaml(cfg))
        # Initialize DataModule
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
        datamodule.prepare_data()
        datamodule.setup()
        log.info(f"DataModule instantiated: {datamodule}")
        log.info(f"Train dataloader: {datamodule.train_dataloader()}")
        log.info(f"Validation dataloader: {datamodule.val_dataloader()}")
        log.info(f"Test dataloader: {datamodule.test_dataloader()}")

    main()
