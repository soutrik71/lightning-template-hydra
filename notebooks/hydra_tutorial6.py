"""
Putting it all together: Configuring your application with a config file and using it in your code with Hydra and OmegaConf
"""

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./configs", config_name="config4")
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.db)
    print(cfg.schema)
    print(cfg.ui)


if __name__ == "__main__":
    my_app()
