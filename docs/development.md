# Development guide

Technical documentation for reproducing the experiments and extending the project.
If you just want to see what the project does, start with the [README](../README.md).

## Requirements

- Python 3.11 (what the report used locally; Kaggle's notebook image runs the same code
  on Python 3.12 with PyTorch preinstalled)

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

### Dataset citation

Jun Cheng, *brain tumor dataset*, figshare (2015),
<https://doi.org/10.6084/m9.figshare.1512427> — 3,064 T1-weighted contrast-enhanced
slices from 233 patients (708 meningioma, 1,426 glioma, 930 pituitary), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The accompanying papers are
Cheng et al., *PLoS ONE* 10(10) (2015) and *PLoS ONE* 11(6) (2016). This project
downloads it through the Kaggle mirror `ashkhagan/figshare-brain-tumor-dataset` and
converts the `.mat` files to PNG; the dataset itself is not redistributed here.

## Reproduce the headline results

Every command below is exactly what produced the runs in the results table (seed 42,
`--amp` on GPU). Run them from the repository root; on Kaggle use the same commands
inside the cell in [Cloud training on Kaggle and Colab](#cloud-training-on-kaggle-and-colab).

```bash
python -m src.dataset.download_data      # once: Figshare -> data/raw/*.png

# Baseline — manual defaults (Phase 2-4 setup)
python -m scripts.train_baseline --epochs 40 --device auto --amp

# A — genetic algorithm (10 individuals x 5 generations = 46 evaluations), then
#     retrain the winning genome (run_optimization prints this exact command)
python -m scripts.run_optimization --epochs 3 --device auto --amp
python -m scripts.train_baseline --epochs 100 --learning-rate 0.000315144 --weight-decay 1.90823e-06 --base-filters 16 --depth 4 --device auto --amp

# Equal-budget random-search reference (same 46 evaluations, same seeds)
python -m scripts.run_random_search --evaluations 46 --epochs 3 --device auto --amp

# E — wider from-scratch U-Net + Focal Tversky + cosine schedule (~78 min on a T4)
python -m scripts.train_baseline --epochs 150 --learning-rate 0.000315144 --weight-decay 1.90823e-06 --base-filters 32 --depth 4 --augmentation advanced_spatial --loss focal_tversky_focal --tversky-alpha 0.6 --tversky-beta 0.4 --lr-schedule cosine --device auto --amp

# F — transfer learning from a pretrained ResNet34 encoder (~47 min on a T4)
python -m scripts.train_baseline --model smp_unet --encoder resnet34 --encoder-weights imagenet --encoder-lr-multiplier 0.1 --epochs 100 --learning-rate 0.000315144 --weight-decay 1.90823e-06 --augmentation advanced_spatial --loss focal_tversky_focal --tversky-alpha 0.6 --tversky-beta 0.4 --lr-schedule cosine --device auto --amp

# Evaluation and figures (use --models to compare any set of checkpoints)
python -m scripts.evaluate_test_set     # pooled + per-slice metrics -> runs/comparison/
python -m scripts.plot_results          # figures and tables -> runs/comparison/
python -m scripts.plot_optimization     # GA convergence figures -> runs/optimization/figures/
```

The Kaggle cell that ran the long jobs (T4 ×2 GPU, Internet on):

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

(Replace the last line with any command from the block above; the GA and random search
run the same way.)

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
`runs/capacity_and_lr_schedule/10_E_ultimate_unet/`. The evaluation and plotting scripts
default to the Phase 4 folders of this layout.

## Optimization analysis

```bash
python -m scripts.run_optimization --epochs 3       # GA -> runs/optimization/best_hyperparameters.json
python -m scripts.run_random_search --evaluations 46  # equal-budget reference -> runs/random_search/
python -m scripts.plot_optimization                 # convergence and budget-comparison figures
```

`scripts.plot_optimization` writes `ga_convergence.png` and `ga_fitness_by_generation.png`
to `runs/optimization/figures/`, plus `budget_comparison.png` once an equal-budget
random-search JSON exists under `runs/random_search/`. `scripts.run_optimization` prints
the exact retraining command for the winning genome; retrain it with
`scripts.train_baseline`. Pass `--evaluations` to the random search equal to the GA's
`fitness_evaluations` (46 for the default 10×5 configuration) so the reference curve is
fair.

## Evaluation

```bash
python -m scripts.evaluate_test_set   # pixel-pooled Dice/IoU + per-slice detection metrics
python -m scripts.plot_results        # figures and tables -> runs/comparison/
```

Besides `test_metrics.csv/json`, the evaluation writes `test_slices.csv` (one row per
slice) and `test_slice_metrics.json` (detection rate, missed tumors and per-slice Dice
percentiles per model), so small-lesion behaviour is visible next to the pooled scores.

## Interface (Streamlit)

An interactive, bilingual (English/Spanish) demo that guides visitors through four
steps: choose a slice, choose the model, look at the four views (preprocessed input,
per-class overlay, predicted mask and per-pixel confidence map) and read a plain-language
interpretation with the detected tumor type, its share of the slice and how confident the
model is. Nine slices from the held-out Testing split ship with the repository under
`app/samples/`, so the app can be tried without uploading anything, while any PNG in
the same domain can also be uploaded. One glioma slice is labelled as a *challenging
case* where the transfer-learning model returns an empty mask.

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

The app ships three checkpoints: `app/models/baseline_unet.pth` (the hand-tuned
reference model, included for comparison), `app/models/e_unet.pth` (model E, custom
U-Net trained from scratch) and `app/models/f_smp_unet_fp16.pth` (model F, U-Net with
a ResNet34 encoder pre-trained on ImageNet, stored in half precision and cast to
float32 at load time). To refresh a checkpoint, copy a trained `runs/.../best_model.pth`
over it:

```bash
cp runs/baseline_vs_ga/01_baseline_manual/best_model.pth app/models/baseline_unet.pth
cp runs/capacity_and_lr_schedule/10_E_ultimate_unet/best_model.pth app/models/e_unet.pth
python -m scripts.export_checkpoint_fp16 --checkpoint runs/transfer_learning/13_F_smp_unet_resnet34/best_model.pth --output app/models/f_smp_unet_fp16.pth
```

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub and make it public (or grant Streamlit access to it).
2. Sign in at <https://share.streamlit.io> with GitHub and click **New app**.
3. Select the repository and the branch that contains the interface (or `develop` once
   it is merged), set the main file to `app/main.py` and deploy. Streamlit installs
   `requirements.txt` automatically; no secrets are required.

## Cloud training on Kaggle and Colab

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

## Tests

```bash
python -m pytest tests
```

## Project structure

```
app/                      bilingual Streamlit demo
  main.py                 UI: guided 4-step flow, no sidebar
  segmentation.py         model loading, preprocessing, inference helpers
  i18n.py                 English / Spanish copy
  models/                 baseline_unet.pth, e_unet.pth and f_smp_unet_fp16.pth
  samples/                example test slices, one labelled "challenging case"
data/raw/                 Training/ and Testing/ PNGs (gitignored, download script)
docs/                     README assets, this guide, report sources
runs/                     experiment artifacts (gitignored): history.csv,
                          checkpoints, figures, diagnostics
scripts/                  CLI entry points (train, optimize, evaluate, plot, export)
src/
  dataset/                Figshare download, patient-wise split, loaders, augmentation
  models/                 custom U-Net, SMP U-Net factory, loss family (Dice-Focal)
  optimization/           hand-coded genetic algorithm, random search, report tables
  utils/                  checkpoint loading, training helpers, plotting, diagnostics
tests/                    pytest suite (dataset, losses, models, optimization, app)
```

## Project timeline

| Phase | What was done | Where to look |
|---|---|---|
| 1 · Data | Figshare download, patient-wise split, PNG conversion | `src/dataset/` |
| 2 · Baseline | U-Net + Dice-Focal loss, training loop, smoke tests | `runs/baseline/` |
| 3 · Genetic algorithm | Hand-coded GA (selection, crossover, mutation) + random search | `src/optimization/` |
| 4 · Fair comparison | Equal-budget budgets, per-class diagnostics | `runs/baseline_vs_ga/` |
| 5 · Loss & augmentation | Focal Tversky, 2×2 augmentation design | `runs/loss_and_augmentation/` |
| 6 · Capacity & LR schedule | Run E: wider U-Net + cosine annealing | `runs/capacity_and_lr_schedule/` |
| 7 · Transfer learning | Run F: pretrained ResNet34 encoder, fp16 export | `runs/transfer_learning/` |
| 8 · Demo & report | Streamlit app, per-slice metrics, figures, documentation | `app/`, `docs/` |

The written report (memoria) accompanies this repository as a separate document; the
assignment it responds to is kept in
[`bases-proyecto-optimizacionyheuristica.md`](../bases-proyecto-optimizacionyheuristica.md).
