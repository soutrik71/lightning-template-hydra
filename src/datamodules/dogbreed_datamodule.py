import os
import lightning as L
from torchvision import transforms
from pathlib import Path
import zipfile
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from loguru import logger
import kaggle
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv, find_dotenv
import rootutils

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


def download_data_kaggle(cfg: DictConfig):
    """Download and unzip the dataset from Kaggle."""
    kaggle_zip = Path(f"{cfg.paths.kaggle_dir.split('/')[-1]}.zip")
    data_dir = Path(cfg.paths.data_dir)

    # Check if Kaggle credentials are set
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        logger.error(
            "Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY in environment."
        )
        raise EnvironmentError("Kaggle credentials are missing.")

    if not kaggle_zip.exists():
        logger.info("Downloading the dataset from Kaggle")
        kaggle.api.dataset_download_files(cfg.paths.kaggle_dir, path="./", unzip=False)

    if not data_dir.exists():
        logger.info(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / kaggle_zip.stem).exists():
        logger.info(f"Unzipping dataset to {data_dir}")
        try:
            with zipfile.ZipFile(kaggle_zip, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        except zipfile.BadZipFile:
            logger.error("Error extracting the zip file. It may be corrupted.")
            raise


def input_dataprep(cfg: DictConfig) -> pd.DataFrame:
    """Prepare the input data by downloading and processing the dataset."""
    download_data_kaggle(cfg)

    dataset_path = (
        Path(cfg.paths.data_dir) / cfg.paths.kaggle_dir.split("/")[-1] / "dataset"
    )
    image_paths = list(dataset_path.glob("*/*.jpg"))

    if len(image_paths) == 0:
        logger.error(f"No images found in dataset directory: {dataset_path}")
        raise FileNotFoundError(f"No images found at {dataset_path}")

    logger.info(f"Total Images: {len(image_paths)}")

    dataset_df = pd.DataFrame(
        {
            "image_path": [str(img) for img in image_paths],
            "label": [img.parent.stem for img in image_paths],
        }
    )

    return dataset_df


class DogbreedDataset(Dataset):
    """Custom Dataset for Dog Breeds."""

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        super().__init__()
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(df["label"].unique()))
        }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        label = self.class_to_idx[self.labels[index]]
        if self.transform:
            image = self.transform(image)
        return image, label


class DogbreedDataModule(L.LightningDataModule):
    """DataModule for Dog Breed Classification."""

    def __init__(self, cfg: DictConfig, dataset_df: pd.DataFrame):
        super().__init__()
        self.cfg = cfg
        self.dataset_df = dataset_df
        self.train_split = cfg.data.train_split
        self.val_split = cfg.data.val_split
        self.test_split = cfg.data.test_split
        self.image_size = cfg.data.image_size
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size

    def _get_transform(self):
        """Return the image transformation pipeline."""
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage: str = None):
        """Split the dataset into training, validation, and test sets."""
        dataset = DogbreedDataset(self.dataset_df, transform=self._get_transform())
        total_size = len(dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def _create_dataloader(self, dataset):
        """Helper to create DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)


@hydra.main(version_base="1.1", config_path="../../configs", config_name="train")
def main_dataloader(cfg: DictConfig):
    # Prepare the dataset
    dataset_df = input_dataprep(cfg)

    # Initialize the data module
    dogbreed_datamodule = DogbreedDataModule(cfg, dataset_df)
    dogbreed_datamodule.setup()

    # DataLoader info
    logger.info("DataModule setup successful")
    train_loader = dogbreed_datamodule.train_dataloader()
    val_loader = dogbreed_datamodule.val_dataloader()

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    return dataset_df, dogbreed_datamodule


if __name__ == "__main__":
    main_dataloader()
