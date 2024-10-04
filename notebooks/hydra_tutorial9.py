import hydra
from omegaconf import DictConfig, OmegaConf


class MyModel:
    def __init__(
        self,
        model_name: str,
        input_dim: int,
        output_dim: int,
        lr: float,
        hidden_dim: int,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.model_name = model_name

    def train(self):
        print(
            f"Training model with input_dim={self.input_dim}, output_dim={self.output_dim}, lr={self.lr}, hidden_dim={self.hidden_dim} and model_name={self.model_name}"
        )


@hydra.main(version_base=None, config_path="configs", config_name="config6")
def run_model(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    # Pass the config to the class constructor
    model = MyModel(
        model_name=cfg.model.name,
        input_dim=cfg.model.params.input_dim,
        output_dim=cfg.model.params.output_dim,
        lr=cfg.model.params.lr,
        hidden_dim=cfg.model.params.hidden_dim,
    )
    model.train()  # Run the class method


if __name__ == "__main__":
    run_model()
