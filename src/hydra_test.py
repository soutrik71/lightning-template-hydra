import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(".env"))

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def hydra_test(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"data test: {cfg.data}")
    print(f"paths test: {cfg.paths}, {cfg.paths.root_dir}, {cfg.paths.data_dir}")
    print(
        f"kaggle info: {cfg.paths.kaggle_dir}, {cfg.paths.kaggle_username}, {cfg.paths.kaggle_key}"
    )


if __name__ == "__main__":
    hydra_test()
