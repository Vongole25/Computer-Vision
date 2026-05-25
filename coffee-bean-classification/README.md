# Defective Coffee Bean Classification with EfficientNet

> 🥈 **2nd Place**, T-SUM Data Analysis Competition · Korea University Sejong (2023) · Team of 4

Image classifier distinguishing **defective** from **normal** coffee beans, built on
**EfficientNet-B0** with a custom preprocessing pipeline (background removal + square
padding + resize). Includes grid search over hyperparameters and full evaluation
(confusion matrix, ROC/AUC).

## TL;DR

- **Task:** Binary image classification — normal vs defective coffee beans.
- **Custom Preprocessing (`utils/img_wrapper.py`):**
  - Background removal via `rembg`
  - **Square padding** to preserve aspect ratio (no distortion)
  - Resize to 224×224 for model input
- **Model:** EfficientNet-B0 with ImageNet-pretrained weights, fine-tuned on the bean dataset.
- **Training:** Grid search over hyperparameters (learning rate, batch size, epochs).
- **Evaluation:** Confusion matrix + Accuracy + ROC curve + AUC.

## Repository Structure

```
coffee-bean-classification/
├── README.md / README_KR.md
├── .gitignore
├── notebooks/
│   └── 01_efficientnet_training.ipynb   Main: model + grid search + evaluation
├── utils/
│   ├── __init__.py
│   └── img_wrapper.py                   Custom preprocessing class
└── _drafts/
    ├── README.md
    ├── preprocessing_test.ipynb         ImgWrapper preview/verification
    └── data_unzipper.ipynb              Dataset extraction helper
```

## Preprocessing Pipeline

```python
from utils import ImgWrapper

wrapper = ImgWrapper(height=224, width=224, channels=4)
processed = wrapper(input_pil_image)
# → background removed → square-padded → resized to 224×224
```

Choices made:
- **Background removal:** Beans were photographed against varying backgrounds.
  `rembg` (U²-Net-based) isolates the bean before downstream features.
- **Square padding (not stretching):** Beans have varying aspect ratios; padding
  preserves the shape signal rather than distorting it.
- **RGBA (4 channels):** Keeps the transparency mask from rembg available.

## Model Training (`notebooks/01_efficientnet_training.ipynb`)

- **Backbone:** EfficientNet-B0, pretrained on ImageNet.
- **Hyperparameter Grid Search:** Iterates over combinations of LR / batch size /
  epochs to pick the best configuration on the held-out set.
- **Evaluation:** Confusion matrix, accuracy, ROC curve, AUC.

## Stack

`Python` · `PyTorch` · `torchvision` (EfficientNet) · `rembg` (onnxruntime) ·
`scikit-learn` · `seaborn` · `Google Colab`

## How to Run

```bash
# Open in Google Colab (preprocessing uses rembg which needs onnxruntime)
# 1. _drafts/data_unzipper.ipynb   (optional) — unzip raw bean datasets
# 2. _drafts/preprocessing_test.ipynb (optional) — sanity-check ImgWrapper
# 3. notebooks/01_efficientnet_training.ipynb — train & evaluate
```

Dataset (bean images, gathered for the competition) is **not** included in this repo
(see `.gitignore`).

## Award

🥈 **2nd Place**, T-SUM Data Analysis Competition, Korea University Sejong (2023) · Team of 4

---

*Author: Sunjae Lee · B.S. Big Data Science @ Korea University Sejong*
*Part of [Computer-Vision](https://github.com/Vongole25/Computer-Vision)*
