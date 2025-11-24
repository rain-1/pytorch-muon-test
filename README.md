## MNIST MLP with PyTorch and Weights & Biases

Train a simple feedforward network on MNIST with AdamW and GELU activations while logging metrics to Weights & Biases.

### Setup
- Install deps: `pip install -r requirements.txt`
- If you need to avoid network access for logging, run with `--wandb-mode offline`.

### Run
```
python train.py --epochs 5 --batch-size 128 --lr 1e-3 --weight-decay 0.01
```
Flags can be adjusted as needed (see `python train.py -h`). Data downloads to `./data` by default.

## Good parameters

`python ./train.py --epochs 8 --batch-size 64 --log-interval 10`
