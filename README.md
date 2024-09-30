# Lightning Hydra Template
- src/settings.py: Configurations for the project (e.g. paths, hyperparameters)
- .env and src/config.yaml 
- src/train.py: Train a model
- src/infer.py: Infer with a model
- From Dockerfile, you can build a docker image and run a container
- docker-compose.yml: You can build a docker image and run a containers for first training and inference upon completion of training
  
## Train a model and Infer Dev

```bash
python src/train.py
```

```bash
python src/infer.py
```

## Docker Compose 

```bash
docker-compose build
docker-compose up
```