## __POETRY SETUP__
```bash
# Install poetry
conda create -n poetry_env python=3.10 -y
conda activate poetry_env
pip install poetry
poetry env info
poetry new pytorch_project
cd pytorch_project/
# fill up the pyproject.toml file without pytorch and torchvision
poetry install

# Add dependencies to the project for pytorch and torchvision
poetry source add --priority explicit pytorch_cpu https://download.pytorch.org/whl/cpu
poetry add --source pytorch_cpu torch torchvision
poetry lock
poetry show

# Add dependencies to the project 
poetry add matplotlib
poetry add hydra-core
poetry add omegaconf
poetry add hydra_colorlog
poetry lock
poetry show
```

## __MULTISTAGEDOCKER SETUP__

#### Step-by-Step Guide to Creating Dockerfile and docker-compose.yml for a New Code Repo

If you're new to the project and need to set up Docker and Docker Compose to run the training and inference steps, follow these steps.

---

### 1. Setting Up the Dockerfile

A Dockerfile is a set of instructions that Docker uses to create an image. In this case, we'll use a **multi-stage build** to make the final image lightweight while managing dependencies with `Poetry`.

#### Step-by-Step Process for Creating the Dockerfile

1. **Choose a Base Image**:
   - We need to choose a Python image that matches the project's required version (e.g., Python 3.10.14).
   - Use the lightweight **`slim`** version to minimize image size.

   ```Dockerfile
   FROM python:3.10.14-slim as builder
   ```

2. **Install Dependencies in the Build Stage**:
   - We'll use **Poetry** for dependency management. Install it using `pip`.
   - Next, copy the `pyproject.toml` and `poetry.lock` files to the `/app` directory to install dependencies.

   ```Dockerfile
   RUN pip3 install poetry==1.7.1
   WORKDIR /app
   COPY pytorch_project/pyproject.toml pytorch_project/poetry.lock /app/
   ```

3. **Configure Poetry**:
   - Configure Poetry to install the dependencies in a virtual environment inside the project directory (not globally). This keeps everything contained and avoids conflicts with the system environment.

   ```Dockerfile
   ENV POETRY_NO_INTERACTION=1 \
       POETRY_VIRTUALENVS_IN_PROJECT=1 \
       POETRY_VIRTUALENVS_CREATE=true \
       POETRY_CACHE_DIR=/tmp/poetry_cache
   ```

4. **Install Dependencies**:
   - Use `poetry install --no-root` to install only the dependencies and not the package itself. This is because you typically don't need to install the actual project code at this stage.

   ```Dockerfile
   RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root
   ```

5. **Build the Runtime Stage**:
   - Now, set up the final runtime image. This stage will only include the required application code and the virtual environment created in the first stage.
   - The final image will use the same Python base image but remain small by avoiding the re-installation of dependencies.

   ```Dockerfile
   FROM python:3.10.14-slim as runner
   WORKDIR /app
   COPY src /app/src
   COPY --from=builder /app/.venv /app/.venv
   ```

6. **Set Up the Path to Use the Virtual Environment**:
   - Update the `PATH` environment variable to use the Python binaries from the virtual environment.

   ```Dockerfile
   ENV PATH="/app/.venv/bin:$PATH"
   ```

7. **Set a Default Command**:
   - Finally, set the command that will be executed by default when the container is run. You can change or override this later in the Docker Compose file.

   ```Dockerfile
   CMD ["python", "-m", "src.train"]
   ```

### Final Dockerfile

```Dockerfile
# Stage 1: Build environment with Poetry and dependencies
FROM python:3.10.14-slim as builder
RUN pip3 install poetry==1.7.1
WORKDIR /app
COPY pytorch_project/pyproject.toml pytorch_project/poetry.lock /app/
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Stage 2: Runtime environment
FROM python:3.10.14-slim as runner
WORKDIR /app
COPY src /app/src
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "src.train"]
```

---

### 2. Setting Up the docker-compose.yml File

The `docker-compose.yml` file is used to define and run multiple Docker containers as services. In this case, we need two services: one for **training** and one for **inference**.

### Step-by-Step Process for Creating docker-compose.yml

1. **Define the Version**:
   - Docker Compose uses a versioning system. Use version `3.8`, which is widely supported and offers features such as networking and volume support.

   ```yaml
   version: '3.8'
   ```

2. **Set Up the `train` Service**:
   - The `train` service is responsible for running the training script. It builds the Docker image, runs the training command, and uses volumes to store the data, checkpoints, and artifacts.

   ```yaml
   services:
     train:
       build:
         context: .
       command: python -m src.train
       volumes:
         - data:/app/data
         - checkpoints:/app/checkpoints
         - artifacts:/app/artifacts
       shm_size: '2g'  # Increase shared memory to prevent DataLoader issues
       networks:
         - default
       env_file:
         - .env  # Load environment variables
   ```

3. **Set Up the `inference` Service**:
   - The `inference` service runs after the training has completed. It waits for a file (e.g., `train_done.flag`) to be created by the training process and then runs the inference script.

   ```yaml
     inference:
       build:
         context: .
       command: /bin/bash -c "while [ ! -f /app/checkpoints/train_done.flag ]; do sleep 10; done; python -m src.infer"
       volumes:
         - checkpoints:/app/checkpoints
         - artifacts:/app/artifacts
       shm_size: '2g'
       networks:
         - default
       depends_on:
         - train
       env_file:
         - .env
   ```

4. **Define Shared Volumes**:
   - Volumes allow services to share data. Here, we define three shared volumes:
     - `data`: Stores the input data.
     - `checkpoints`: Stores the model checkpoints and the flag indicating training is complete.
     - `artifacts`: Stores the final model outputs or artifacts.

   ```yaml
   volumes:
     data:
     checkpoints:
     artifacts:
   ```

5. **Set Up Networking**:
   - Use the default network to allow the services to communicate.

   ```yaml
   networks:
     default:
   ```

### Final docker-compose.yml

```yaml
version: '3.8'

services:
  train:
    build:
      context: .
    command: python -m src.train
    volumes:
      - data:/app/data
      - checkpoints:/app/checkpoints
      - artifacts:/app/artifacts
    shm_size: '2g'
    networks:
      - default
    env_file:
      - .env

  inference:
    build:
      context: .
    command: /bin/bash -c "while [ ! -f /app/checkpoints/train_done.flag ]; do sleep 10; done; python -m src.infer"
    volumes:
      - checkpoints:/app/checkpoints
      - artifacts:/app/artifacts
    shm_size: '2g'
    networks:
      - default
    depends_on:
      - train
    env_file:
      - .env

volumes:
  data:
  checkpoints:
  artifacts:

networks:
  default:
```

---

### Summary

1. **Dockerfile**:
   - A multi-stage Dockerfile is used to create a lightweight image where the dependencies are installed with Poetry and the application code is run using a virtual environment.
   - It ensures that all dependencies are isolated in a virtual environment, and the final container only includes what is necessary for the runtime.

2. **docker-compose.yml**:
   - The `docker-compose.yml` file defines two services:
     - **train**: Runs the training script and stores checkpoints.
     - **inference**: Waits for the training to finish and runs inference based on the saved model.
   - Shared volumes ensure that the services can access data, checkpoints, and artifacts.
   - `shm_size` is increased to prevent issues with DataLoader in PyTorch when using multiple workers.

This setup allows for easy management of multiple services using Docker Compose, ensuring reproducibility and simplicity.

## **References**

- <https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker>
- <https://github.com/fralik/poetry-with-private-repos/blob/master/Dockerfile>
- <https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0>
- <https://www.martinrichards.me/post/python_poetry_docker/>
- <https://gist.github.com/soof-golan/6ebb97a792ccd87816c0bda1e6e8b8c2>

8. ## **DVC SETUP**

dvc init
dvc add data
dvc remote add -d myremote /tmp/dvcstore
dvc push
