# Two-Level Optimization for Automatic Brain Tumor Segmentation in Magnetic Resonance Imaging

### Gradient descent in a U-Net and meta-optimization of hyperparameters with a genetic algorithm

**Javier Ruano Hernández · Alejandro Delgado Valera**
University of Las Palmas de Gran Canaria · Course *Optimization and Heuristics* · Academic year 2026–2027

**Language:** [Español](README.md) · **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](docs/development.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](docs/development.md)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#demo)

![Input slice, model F prediction and expert annotation](docs/hero_overlay.png)

*Left: input slice · Center: prediction of model F · Right: the expert's annotation (this glioma slice: Dice 0.87).*

## About this project

Segmenting brain tumors in an MRI scan is, for a neural network, a hard problem for two reasons: the tumor covers on average just **1.7 % of the pixels** of each slice (a strong class imbalance), and an expert's annotation is not exact to the millimeter. This project tackles the challenge by framing it as a **two-level optimization problem**:

- **Inner level — training the network well.** Training a U-Net is formulated as minimizing a non-convex loss function over 1.8 to 7.2 million parameters with stochastic gradient descent (Adam). From the optimization viewpoint we analyze the architecture, the surrogate loss functions (Dice, Focal and Focal Tversky) and the learning-rate schedule.
- **Outer level — tuning the hyperparameters.** The four hyperparameters that condition that training (learning rate, weight decay, number of filters and depth) are set by a **hand-written genetic algorithm** with real-valued encoding, tournament selection, BLX-α crossover, gaussian mutation with reflection and elitism, and a low-fidelity fitness function: the validation Dice after only three training epochs.

With 46 evaluations, the genetic algorithm improves the fitness of the manual configuration from 0.441 to 0.509 and, retrained at full length, the evolved configuration raises the mean test Dice from **0.691 to 0.776**. A posterior analysis of the heuristic solution reveals a *short-horizon bias* in the fitness function —it favors small, fast-converging networks—, which we correct with extra capacity, a cosine learning-rate schedule and a diagnosis-driven loss calibration: the final from-scratch model reaches **0.8175**, and with transfer learning from an ImageNet-pretrained ResNet34, **0.8250** —**0.134** above the starting point. The genetic algorithm alone accounts for roughly **two thirds of the total improvement**.

Everything is validated with a *patient-wise* split on a public dataset of 3,064 slices, and the project is completed with a **bilingual interactive demo** that anyone can run in the browser.

## Demo

> **Live demo link:** [streamlit.app](https://unet-tumor-segmentation-jruano-adelgado.streamlit.app/)

## How it works

1. **Patient-wise data split.** 3,064 contrast-enhanced T1 slices from 233 patients (Figshare dataset), split into 2,004 for training, 468 for validation and 592 for testing. The split is by patient, not by image: no patient appears in both training and test, and the test set is **never** used to pick models or hyperparameters.

2. **A U-Net that classifies every pixel.** The usual encoder–decoder architecture in medical segmentation: every pixel of the image is classified as background, meningioma, glioma or pituitary tumor.

3. **Losses designed for class imbalance.** Since the tumor covers a tiny fraction of the image, a plain per-pixel loss would learn to predict «background» every time. We combine the Dice coefficient (overlap with the expert's annotation) with the Focal and Focal Tversky losses, which penalize false negatives harder, and calibrate their weights with an error analysis (per-class precision/recall and confusion matrices).

4. **Data augmentation and learning-rate scheduling.** A 2×2 factorial design of augmentation (spatial / intensity) and a comparison of constant versus cosine learning rates, with model selection by early stopping on the validation set.

5. **Outer-loop optimization with a genetic algorithm.** A real-valued GA tunes learning rate, weight decay, filters and depth. Fitness is evaluated at low fidelity (validation Dice at three epochs) with memoization, enabling **46 evaluations** in total, and benchmarked against an **equal-budget random search** as a fair reference. The result is also analyzed critically: the GA converges 3.2× faster than the random search, but its fitness suffers from the short-horizon bias, a defect we identify and fix in the refinement phase.

6. **Transfer learning as a complement.** The best from-scratch model is compared against a U-Net whose ResNet34 encoder is pretrained on ImageNet (with a reduced learning rate for the encoder): it reduces confusion between tumor types, without a statistically significant gain in delineation (patient-level bootstrap confidence intervals).

7. **Rigorous evaluation.** Per-class Dice and IoU, per-slice detection rate (how many small tumors slip through), false positive/negative analysis and controlled pairwise retraining comparisons. Checkpoints are selected on validation and evaluated exactly once on test.

## Dataset

Jun Cheng, *brain tumor dataset*, figshare (2015), <https://doi.org/10.6084/m9.figshare.1512427>
— 3,064 contrast-enhanced T1 slices from 233 patients (708 meningioma, 1,426 glioma, 930 pituitary), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The accompanying papers are Cheng et al., *PLoS ONE* 10(10) (2015) and 11(6) (2016). The dataset is downloaded through the Kaggle mirror `ashkhagan/figshare-brain-tumor-dataset` and is **not redistributed** in this repository.

## Repository layout

```
app/        bilingual Streamlit demo (Spanish/English) with the baseline, E and F
src/        data pipeline, U-Net networks, loss functions and genetic algorithm
scripts/    training, optimization, evaluation and plotting
tests/      pytest test suite
docs/       technical guide, README figures
```

## Further reading

- **[docs/development.md](docs/development.md)** — full technical guide: installation, reproducing every result, training options, Streamlit Cloud deployment and Kaggle/Colab execution.
- The **full report (memoria)** accompanies this submission as a separate document.


## License

The code in this repository is released under the [MIT License](LICENSE). The dataset is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by its original authors and is **not** redistributed here.

## Medical disclaimer

This is a research and teaching prototype. It is **not** a medical device, it has not been clinically validated, and it must never be used to diagnose, treat or rule out disease. All reported numbers are reproducible experiments on a public dataset.
