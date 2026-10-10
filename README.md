# optimized-neural-network-tumor-segmentation

Automatic brain tumor segmentation on MRI scans using a modular U-Net with a
Dice-Focal loss, tuned with metaheuristic optimization (Genetic Algorithms / PSO).

## Requirements

- Python 3.11

## Installation

### CPU

```bash
pip install -r requirements.txt
```

### NVIDIA GPU (CUDA)

The default wheels of `torch`/`torchvision` already include CUDA support, so the
command above also works on GPU machines. Verify it:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If it prints `False`, reinstall PyTorch with the wheel that matches your driver
(pick your CUDA version at https://pytorch.org/get-started/locally/):

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
```

## Dataset

```bash
python -m src.dataset.download_data   # downloads Figshare from Kaggle and converts it to PNG
python -m scripts.visualize_sample    # optional: visual sanity check
```

Alternatively, copy a prepared `data/raw/Training` and `data/raw/Testing` from
another machine (the patient-wise split is seed-based, so it stays identical).

## Training

```bash
python -m scripts.train_baseline --epochs 40 --device auto           # CPU or GPU
python -m scripts.train_baseline --epochs 40 --device auto --amp     # GPU with mixed precision
python -m scripts.train_baseline --epochs 2 --limit-batches 20       # quick smoke test
```

Artifacts are written to `runs/baseline/` (`history.csv` and `best_model.pth`).

## Tests

```bash
python -m pytest tests
```
