from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./configs", config_name="config3")
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
