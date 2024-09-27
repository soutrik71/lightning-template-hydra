# Lightning Hydra Template

## Train

Dev Run

```bash
python src/train.py experiment=catdog_ex +trainer.fast_dev_run=True
```

```bash
python src/train.py experiment=catdog_ex +trainer.log_every_n_steps=5
```
