class MyModel:
    """
    A simple class to demonstrate how to use Hydra with classes
    """

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
