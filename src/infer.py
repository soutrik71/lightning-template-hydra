from pathlib import Path
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.dogbreed_classifier import DogbreedClassifier
from src.utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
import pandas as pd
from loguru import logger
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv, find_dotenv
import rootutils

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root")


@task_wrapper
def load_image(image_path: str):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img, transform(img).unsqueeze(0)


@task_wrapper
def infer(model: torch.nn.Module, image_tensor: torch.Tensor, classes: list):
    """Perform inference on the provided image tensor."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = classes[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence


@task_wrapper
def save_prediction_image(
    image: Image.Image, predicted_label: str, confidence: float, output_path: Path
):
    """Save the image with the prediction overlay."""
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the output directory exists
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@task_wrapper
def download_image(cfg: DictConfig):
    """Download an image from the web for inference."""
    url = "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&w=600"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code == 200:
        image_path = Path(cfg.paths.root_dir) / "dog.jpg"
        with open(image_path, "wb") as file:
            file.write(response.content)
        logger.info(f"Image downloaded successfully as {image_path}!")
    else:
        logger.error(f"Failed to download image. Status code: {response.status_code}")


@hydra.main(config_path="../configs", config_name="infer", version_base="1.1")
def main_infer(cfg: DictConfig):
    logger_path = Path(cfg.paths.log_dir) / "infer.log"
    setup_logger(logger_path)

    # Remove the train_done flag if it exists
    flag_file = Path(cfg.paths.ckpt_dir) / "train_done.flag"
    if flag_file.exists():
        flag_file.unlink()

    # Load the trained model
    logger.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = DogbreedClassifier.load_from_checkpoint(checkpoint_path=cfg.ckpt_path)

    # Download an image for inference
    logger.info("Downloading an image for inference")
    download_image(cfg)

    output_folder = Path(cfg.paths.artifact_dir)
    classes = (
        pd.read_csv(Path(cfg.paths.artifact_dir) / "dogbreed_dataset.csv")["label"]
        .unique()
        .tolist()
    )

    logger.info("Starting inference on the downloaded image")
    image_files = [
        f
        for f in Path(cfg.paths.root_dir).iterdir()
        if f.suffix in {".jpg", ".jpeg", ".png"}
    ]

    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for image_file in image_files:
            img, img_tensor = load_image(image_file)
            predicted_label, confidence = infer(
                model, img_tensor.to(model.device), classes
            )

            output_file = output_folder / f"{image_file.stem}_prediction.png"
            logger.info(f"Saving prediction images to {output_file}")
            save_prediction_image(img, predicted_label, confidence, output_file)

            progress.console.print(
                f"Processed {image_file}: {predicted_label} ({confidence:.2f})"
            )
            progress.advance(task)

            logger.info(f"Processed {image_file}: {predicted_label} ({confidence:.2f})")


if __name__ == "__main__":
    main_infer()
