# Lightning Hydra Project Template

## Train

```bash
python src/train.py
```

## Tensorboard

```bash
tensorboard --logdir logs
```

## Infer

```bash
python src/infer.py --input_folder samples --output_folder predictions --ckpt_path "/workspace/lightning-template-hydra/logs/catdog_classification/version_5/checkpoints/epoch=0-step=3.ckpt"
```

## Black

```bash
black .
```
