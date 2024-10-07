# Stage 1: Build environment with Poetry and dependencies
FROM python:3.10.14-slim as builder

# Install Poetry
RUN pip3 install poetry==1.7.1

# Set the working directory to /app
WORKDIR /app

# Copy pyproject.toml and poetry.lock to install dependencies
COPY pytorch_project/pyproject.toml pytorch_project/poetry.lock /app/

# Configure Poetry environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install dependencies without installing the package itself (use --no-root)
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Stage 2: Runtime environment
FROM python:3.10.14-slim as runner

# Set the working directory to /app
WORKDIR /app

# Copy application source code
COPY src /app/src

# copy project configs
COPY configs /app/configs

# copy the .project-root file
COPY .project-root /app/.project-root

# Copy hello.py
COPY hello.py /app/hello.py

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Set the environment path to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set the default command for the container (this can also be set in Docker Compose)
CMD ["python", "-m", "hello"]
