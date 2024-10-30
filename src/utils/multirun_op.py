import os
import glob
import yaml
import pandas as pd
import json
from datetime import datetime
from loguru import logger


def multirun_artifact_producer(base_path: str, output_path: str):
    """Aggregate data from multiple runs and save to a JSON file."""
    latest_folder = max(glob.glob(os.path.join(base_path, "*")), key=os.path.getmtime)
    if len(latest_folder) == 0:
        logger.error("No run folders found!")
        return

    # Initialize JSON structure
    output_data = {}
    # Process each run directory
    for run_dir in os.listdir(latest_folder):
        run_path = os.path.join(latest_folder, run_dir)
        if os.path.isdir(run_path):
            # Paths to files
            hparams_path = os.path.join(run_path, "csv", "version_0", "hparams.yaml")
            metrics_path = os.path.join(run_path, "csv", "version_0", "metrics.csv")

            # Read hparams.yaml
            with open(hparams_path, "r") as file:
                hparams = yaml.safe_load(file)

            # Read metrics.csv and calculate averages
            metrics_df = pd.read_csv(metrics_path)
            avg_train_acc = metrics_df["train_acc"].dropna().mean()
            avg_val_acc = metrics_df["val_acc"].dropna().mean()

            # Create JSON structure for this run
            output_data[f"run{run_dir}"] = {
                "hparams": hparams,
                "metrics": {"avg_train_acc": avg_train_acc, "avg_val_acc": avg_val_acc},
            }

    # Save to JSON
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path, f"aggregated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    logger.info(f"Saving aggregated data to {output_file}")
    with open(output_file, "w") as json_file:
        json.dump(output_data, json_file, indent=4)


if __name__ == "__main__":
    # Paths
    base_path = "./logs/train/multiruns"
    output_path = "./artifacts"
    multirun_artifact_producer(base_path, output_path)
