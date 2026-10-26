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

The app loads `app/models/baseline_unet.pth`. Once the genetic search produces its best
checkpoint, copy it into the app and the selector will offer it too:

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
!pip install -q numpy opencv-python h5py kaggle scikit-learn scipy matplotlib pandas pytest
```

If the repository is private, clone it with a token or make it public (see the
"Access to a private repository" note below).

### Kaggle Notebooks (recommended for long runs)

1. New notebook → Settings → **Accelerator** `T4 x2` (or `P100`) and **Internet** on.
2. Run these cells in order (the Kaggle CLI is already authenticated, so the
   dataset needs no token; cloning into `/kaggle/temp` keeps the dataset out of
   the saved output):

```bash
%cd /kaggle/temp
!git clone https://github.com/oh-j-ruano-a-delgado/optimized-neural-network-tumor-segmentation.git
%cd optimized-neural-network-tumor-segmentation
!pip install -q numpy opencv-python h5py kaggle scikit-learn scipy matplotlib pandas pytest
!python -m src.dataset.download_data
!python -m scripts.train_baseline --epochs 40 --device auto --amp
!cp -r runs/baseline /kaggle/working/
```

Only `/kaggle/working` survives the session, hence the final copy. To run without
keeping the tab open, use **Save & Run All (Commit)** (up to 12 hours).

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
