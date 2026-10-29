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

Phase 5 options (defaults reproduce the Phase 2-4 setup):

```bash
python -m scripts.train_baseline --augmentation advanced --loss focal_tversky_focal   # Focal Tversky (alpha 0.7, beta 0.3, gamma 4/3) + Focal
python -m scripts.train_baseline --loss dice_ce                                        # Dice + cross-entropy control
python -m scripts.train_baseline --augmentation advanced_spatial --loss focal_tversky_focal --epochs 200 --early-stopping-patience 20
python -m scripts.evaluate_diagnostics --checkpoint runs/baseline_vs_ga/02_A_ga_optimized/best_model.pth     # per-class precision/recall and confusion matrices on the validation split
```

Augmentation presets form a 2 x 2 design: `basic` (basic spatial), `advanced_spatial` (advanced
spatial only), `advanced_intensity` (basic spatial + intensity) and `advanced` (both). With
`--early-stopping-patience N`, training stops after N epochs without a new best validation Dice.
With `--lr-schedule cosine`, the learning rate anneals to zero over `--epochs` epochs.

Phase 7 options (transfer learning with `segmentation_models_pytorch`):

```bash
python -m scripts.train_baseline --model smp_unet --encoder resnet34 --encoder-lr-multiplier 0.1   # ImageNet-pretrained ResNet34 encoder, encoder at 1/10 of --learning-rate
python -m scripts.train_baseline --model smp_unet --encoder resnet34 --encoder-weights none        # same network from random initialization (pretraining control)
```

`--model unet` (default) keeps the custom U-Net. The ImageNet weights are downloaded from the
Hugging Face Hub the first time an SMP model is built for training, so that step needs internet
access; evaluating a saved checkpoint does not.

Local experiment archive (`runs/` is gitignored): runs are grouped by phase and numbered in
chronological order, e.g. `runs/baseline_vs_ga/02_A_ga_optimized/` or
`runs/phase6_capacity_and_lr_schedule/10_E_ultimate_unet/`. The evaluation and plotting scripts
default to the Phase 4 folders of this layout.

## Interface (Streamlit)

An interactive, bilingual (English/Spanish) demo that guides visitors through three
steps: choose a slice, look at the four views (preprocessed input, per-class overlay,
predicted mask and per-pixel confidence map) and read a plain-language interpretation
with the detected tumor type, its share of the slice and how confident the model is.
Nine slices from the held-out Testing split ship with the repository under
`app/samples/`, so the app can be tried without uploading anything, while any PNG in
the same domain can also be uploaded. Technical controls (checkpoint selection, overlay
opacity, confidence threshold) live in a collapsed "Advanced settings" section.

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

The app ships two checkpoints: `app/models/baseline_unet.pth` (manually tuned
baseline) and `app/models/optimized_unet.pth` (genetic-algorithm winner). Both appear in
the "Advanced settings" checkpoint selector, which defaults to the optimized one. To
refresh a checkpoint, copy a trained `runs/.../best_model.pth` over it:

```bash
cp runs/optimized/best_model.pth app/models/optimized_unet.pth
```

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub and make it public (or grant Streamlit access to it).
2. Sign in at <https://share.streamlit.io> with GitHub and click **New app**.
3. Select the repository and the branch that contains the interface (or `develop` once
   it is merged), set the main file to `app/main.py` and deploy. Streamlit installs
    `requirements.txt` automatically; no secrets are required.

## Cloud training (Kaggle / Colab)

Both platforms ship PyTorch with CUDA preinstalled, so install only the remaining
dependencies (this avoids re-downloading several GB of torch wheels):

```bash
!pip install -q numpy opencv-python h5py kaggle scikit-learn scipy matplotlib pandas pytest albumentations==2.0.8 segmentation-models-pytorch==0.6.0
```

If the repository is private, clone it with a token or make it public (see the
"Access to a private repository" note below).

### Kaggle Notebooks (recommended for long runs)

1. New notebook → Settings → **Accelerator** `T4 x2` (or `P100`) and **Internet** on.
2. Clone the code **inside `/kaggle/working`** (the only directory that survives the
   session, so every artifact is kept in the Output tab). `main` only contains the README,
   so pass the branch that holds the code with `--branch` (no need to change the
   repository default branch):

```bash
%%bash
set -e
cd /kaggle/working
git clone --branch develop --single-branch \
  https://github.com/oh-j-ruano-a-delgado/optimized-neural-network-tumor-segmentation.git
cd optimized-neural-network-tumor-segmentation
pip install -q numpy opencv-python h5py kaggle scikit-learn scipy matplotlib pandas pytest albumentations==2.0.8 segmentation-models-pytorch==0.6.0
python -m src.dataset.download_data
python -m scripts.train_baseline --epochs 40 --device auto --amp
```

3. Hyperparameter search (genetic algorithm and its equal-budget random-search
   reference). Both write their history JSON under `runs/`, so download them from the
   notebook Output before the session ends:

```bash
%%bash
set -e
cd /kaggle/working/optimized-neural-network-tumor-segmentation
python -m scripts.run_optimization --epochs 3 --device auto --amp
python -m scripts.run_random_search --evaluations 46 --epochs 3 --device auto --amp
```

The default GA configuration (10 individuals, 5 generations, one elite) costs 46 fitness
evaluations, so the random-search reference uses the same 46. `run_optimization` prints
the command that retrains the winning configuration for the final model; it adds about
one hour. To run without keeping the tab open, use **Save & Run All (Commit)**
(12-hour limit per session, ~30 GPU-hours per week).

### Google Colab (interactive debugging)

Same `git clone` and `pip install` cells, but the Kaggle API needs your token:

```python
from google.colab import files
files.upload()   # upload kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!python -m src.dataset.download_data
```

Colab's disk is ephemeral, so save the artifacts to Google Drive when done:

```python
!python -m scripts.train_baseline --epochs 40 --device auto --amp
from google.colab import drive
drive.mount('/content/drive')
!cp -r runs/baseline /content/drive/MyDrive/
```

### Access to a private repository

`git clone` needs credentials when the repository is private. On Kaggle, store a
GitHub token as a notebook secret (Add-ons → Secrets) and use it:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/oh-j-ruano-a-delgado/optimized-neural-network-tumor-segmentation.git
```

## Hyperparameter optimization and comparison

```bash
python -m scripts.run_optimization --epochs 3                             # GA -> runs/optimization/best_hyperparameters.json
python -m scripts.run_random_search --evaluations 46                      # equal-budget reference -> runs/random_search/
python -m scripts.evaluate_test_set                                       # baseline vs optimized on the Testing split
python -m scripts.plot_results                                            # figures and tables -> runs/comparison/
```

The winning genome is retrained for the final model with `scripts.train_baseline` (the
optimization script prints the exact command). Pass `--evaluations` to the random search
equal to the GA's `fitness_evaluations` (46 for the default configuration) for a fair
reference curve.

## Tests

```bash
python -m pytest tests
```
