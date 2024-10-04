from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="./configs", config_name="config2")
def my_app(cfg: DictConfig):
    print(cfg)
    print(cfg.node.loompa)
    print(cfg.node.zippity)
    print(cfg.node.do)
    # print(cfg.node.waldo)


if __name__ == "__main__":
    my_app()
