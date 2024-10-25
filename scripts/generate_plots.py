import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Find the most recent metrics.csv file
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

# Create training loss plot
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train_loss"], label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.savefig("train_loss.png")
plt.close()

# Create training accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train_acc"], label="Training Accuracy")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Steps")
plt.legend()
plt.savefig("train_acc.png")
plt.close()

# Generate test metrics table
test_metrics = df.iloc[-1]
test_table = "| Metric | Value |\n|--------|-------|\n"
test_table += f"| Test Accuracy | {test_metrics['val_acc']:.4f} |\n"
test_table += f"| Test Loss | {test_metrics['val_loss']:.4f} |\n"

# Write the test metrics table to a file
with open("test_metrics.md", "w") as f:
    f.write(test_table)

print("Plots and test metrics table generated successfully.")
