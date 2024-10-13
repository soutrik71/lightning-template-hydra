Here is a README for your GitHub repository based on the provided GitHub Actions (`ci.yml`) file and `docker-compose.yml` file:

---

# Project: Lightning Template with Hydra

This repository contains a project template that uses [PyTorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/) for training machine learning models. The project includes a fully configured CI/CD pipeline using GitHub Actions for testing, linting, and deploying the Docker images, as well as a `docker-compose` setup for managing different stages of the model lifecycle (train, eval, inference).

## Table of Contents
- [Project: Lightning Template with Hydra](#project-lightning-template-with-hydra)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Installation](#installation)
  - [GitHub Actions CI/CD Pipeline](#github-actions-cicd-pipeline)
    - [Jobs Overview](#jobs-overview)
    - [CI/CD Artifacts](#cicd-artifacts)
  - [Docker Compose Setup](#docker-compose-setup)
    - [Services Overview](#services-overview)
    - [Volumes](#volumes)
  - [How to Use](#how-to-use)
    - [Running Locally](#running-locally)
    - [Running via Docker Compose](#running-via-docker-compose)

## Setup

### Requirements
- Python 3.10.x
- Poetry for dependency management
- Docker and Docker Compose

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lightning-template-hydra.git
   cd lightning-template-hydra
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create a `.env` file in the root directory to store your environment variables (e.g., Kaggle API keys):
   ```bash
   KAGGLE_USERNAME=<your-kaggle-username>
   KAGGLE_KEY=<your-kaggle-key>
   ```

## GitHub Actions CI/CD Pipeline

The CI pipeline is defined in the `ci.yml` file and triggers on `push` and `pull_request` events to the `main` and `feat/*` branches. It includes the following key steps:

### Jobs Overview

- **`python_basic_test`**: 
  - Sets up the Python environment using the specified version (`3.10.14`).
  - Installs dependencies with Poetry.
  - Runs tests with `pytest` and collects code coverage.
  - Lints the code using `flake8`.
  - Runs static type checks with `mypy`.

- **`pytorch_code_test`**:
  - Runs the PyTorch model training (`python -m src.train`) and evaluates model performance.
  - Fails the build if the training accuracy falls below a threshold of 95%.
  - Uploads model checkpoints, logs, and config files as GitHub Actions artifacts for further analysis.

- **`build-and-push-image`**:
  - Builds and pushes the Docker image to the GitHub Container Registry using the repository’s metadata (tags, labels).

### CI/CD Artifacts

Artifacts generated during the GitHub Actions pipeline include:
- **Model Checkpoints**: Stored in the `checkpoints/` directory.
- **Model Logs**: Stored in the `logs/` directory.
- **Config Files**: Stored in the `configs/` directory.
  
These files can be retrieved from the "Artifacts" section of the corresponding workflow run in GitHub Actions.

## Docker Compose Setup

The `docker-compose.yml` file defines the following services: **train**, **eval**, and **inference**, which use the shared data, checkpoints, artifacts, and logs directories.

### Services Overview

- **`train`**: 
  - Builds the image and runs the model training (`python -m src.train`).
  - Volumes:
    - `data:/app/data`
    - `checkpoints:/app/checkpoints`
    - `artifacts:/app/artifacts`
    - `logs:/app/logs`

- **`eval`**:
  - Waits for a flag (`train_done.flag`) from the training service before starting evaluation (`python -m src.eval`).
  - Volumes and dependencies are the same as the `train` service.

- **`inference`**:
  - Waits for the `train_done.flag` before running inference (`python -m src.infer`).
  - Uses the same volumes and dependencies as `train`.

### Volumes

The following shared volumes are used:
- `data`: Used for dataset storage.
- `checkpoints`: Used for model checkpoints.
- `artifacts`: Used for storing any model artifacts generated during training and evaluation.
- `logs`: Used for logging during all stages of training, evaluation, and inference.

## How to Use

### Running Locally

To run the code locally, make sure all dependencies are installed and that you have the `.env` file set up with the necessary secrets (like Kaggle credentials).

1. Activate the virtual environment:
   ```bash
   poetry shell
   ```

2. Run the model training:
   ```bash
   python -m src.train
   ```

3. After training is completed, you can manually run the evaluation and inference stages:
   ```bash
   python -m src.eval
   python -m src.infer
   ```

### Running via Docker Compose

To run the services using Docker Compose:

1. Build and start the training process:
   ```bash
   docker-compose up -d train
   ```

2. Once training completes, evaluation and inference will automatically start when the `train_done.flag` is detected:
   ```bash
   docker-compose up -d eval
   docker-compose up -d inference
   ```

3. To bring down all services:
   ```bash
   docker-compose down
   ```