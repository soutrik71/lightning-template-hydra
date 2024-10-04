import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from notebooks.models import MyModel


@hydra.main(version_base=None, config_path="configs", config_name="config7")
def run_model(cfg: DictConfig):
    # Dynamically instantiate the class from the config
    model = instantiate(cfg.model)
    model.train()


if __name__ == "__main__":
    run_model()
