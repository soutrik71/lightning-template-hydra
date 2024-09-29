import os
from pathlib import Path
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.dogbreed_classifier import DogbreedClassifier
from src.utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
from src.settings import settings
import pandas as pd
from loguru import logger


@task_wrapper
def load_image(image_path):
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
def infer(model, image_tensor, classes):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map the predicted class to the label
    class_labels = classes
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence


@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@task_wrapper
def download_image():
    url = "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&w=600"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(url, headers=headers, allow_redirects=True)

    if response.status_code == 200:
        with open("./dog.jpg", "wb") as file:
            file.write(response.content)
        print("Image downloaded successfully as dog.jpg!")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


@task_wrapper
def main():
    # Load the trained model
    logger.info(
        f"Loading model from checkpoint: {settings.train_config.checkpoint_path}"
    )
    model = DogbreedClassifier.load_from_checkpoint(
        settings.train_config.checkpoint_path
    )
    # download an image for inference
    logger.info("Downloading an image for inference")
    download_image()
    output_folder = Path("model_output")
    output_folder.mkdir(exist_ok=True, parents=True)

    classes = (
        pd.read_csv(
            os.path.join(settings.data_config.artifact_path, "dogbreed_dataset.csv")
        )["label"]
        .unique()
        .tolist()
    )

    logger.info("starting inference on the downloaded image")
    image_files = os.listdir(".")
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for image_file in image_files:
            if (
                image_file.endswith(".jpg")
                or image_file.endswith(".jpeg")
                or image_file.endswith(".png")
            ):
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(
                    model, img_tensor.to(model.device), classes
                )

                output_file = (
                    output_folder / f"{image_file.split('.')[0]}_prediction.png"
                )
                save_prediction_image(img, predicted_label, confidence, output_file)

                progress.console.print(
                    f"Processed {image_file}: {predicted_label} ({confidence:.2f})"
                )
                progress.advance(task)


if __name__ == "__main__":
    setup_logger("logs/infer_log.log")
    main()
