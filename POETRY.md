## __Installation__
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
```