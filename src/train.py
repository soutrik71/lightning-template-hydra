import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback
import rootutils
from typing import List, Optional
import os
from pathlib import Path

# Set up the root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports after setting up the root
from src.utils.logging_utils import setup_logger, task_wrapper, logger, log_metrics_table

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from config."""
    callbacks: List[Callback] = []
    
    if not callbacks_cfg:
        return callbacks

    for cb_name, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_name}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiate loggers from config."""
    loggers: List[Logger] = []

    if not logger_cfg:
        return loggers
    
    if isinstance(logger_cfg, DictConfig):
        for lg_name, lg_conf in logger_cfg.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                logger.info(f"Instantiating logger <{lg_name}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

def get_checkpoint_path(config: DictConfig) -> Optional[str]:
    """Get the checkpoint path for resuming training or testing."""
    if config.get("ckpt_path"):
        return config.ckpt_path
    
    if config.get("callbacks") and "model_checkpoint" in config.callbacks:
        ckpt_dir = config.callbacks.model_checkpoint.get("dirpath", "")
        if ckpt_dir and os.path.exists(ckpt_dir):
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if ckpt_files:
                return os.path.join(ckpt_dir, ckpt_files[-1])  # Return the last checkpoint
    
    return None

@task_wrapper
def train_and_test(config: DictConfig) -> None:
    """Train and optionally test the model."""
    logger.info("Initializing data module")
    data_module = hydra.utils.instantiate(config.data)

    logger.info("Initializing model")
    model = hydra.utils.instantiate(config.model)

    logger.info("Initializing callbacks")
    callbacks = instantiate_callbacks(config.get("callbacks"))

    logger.info("Initializing loggers")
    loggers = instantiate_loggers(config.get("logger"))
    
    logger.info("Initializing trainer")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers
    )

    ckpt_path = get_checkpoint_path(config)
    if ckpt_path:
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        logger.info("Starting training from scratch")

    logger.info("Starting training")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    logger.info("Training completed")
    log_metrics_table(trainer.callback_metrics, "Training Metrics")

    # Test if requested
    if config.test:
        logger.info("Starting testing")
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Using best checkpoint: {best_model_path}")
        else:
            logger.warning("No best checkpoint found. Using current model weights.")
            best_model_path = None
        
        trainer.test(model=model, datamodule=data_module, ckpt_path=best_model_path)
        logger.info("Testing completed")
        log_metrics_table(trainer.callback_metrics, "Test Metrics")

@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(config: DictConfig) -> None:
    # Set up logger
    setup_logger(Path(config.paths.output_dir) / "train.log")

    # Train and optionally test the model
    train_and_test(config)

if __name__ == "__main__":
    main()