"""
hydra multi run either from the command line using --multirun (-m) or from a config file using the hydra.sweeper.params
python my_app.py --multirun db=mysql,postgresql schema=warehouse,support,school
"""

from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path="configs", config_name="config5")
def my_app(cfg: DictConfig) -> None:
    print(f"driver={cfg.db.driver}, timeout={cfg.db.timeout}")
    print(f"schema={cfg.schema.database}")


if __name__ == "__main__":
    my_app()
