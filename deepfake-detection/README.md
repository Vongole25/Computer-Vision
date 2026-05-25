# Deepfake Detection — Xception+LSTM vs Video Swin Transformer

> 🥉 **3rd Place**, T-SUM Data Analysis Competition · Korea University Sejong (2023) · Team of 4

End-to-end deepfake video classification combining a **face-detection preprocessing pipeline**
with two competing deep-learning architectures: **Xception + LSTM** (frame-level CNN + temporal RNN)
and **Video Swin Transformer** (3D self-attention).

## TL;DR

- **Task:** Binary classification — *real* vs *manipulated* video.
- **Pipeline:**
  1. Face crop & alignment with **RetinaFace** on each frame
  2. Two models trained and compared:
     - **Xception + LSTM** — CNN feature extraction per frame, LSTM over frame sequence
     - **Video Swin Transformer** — 3D shifted-window self-attention on video volume
  3. Evaluation: frame-level + **video-level (multi-face aggregation)**, confusion matrix
- **Environment:** Google Colab + GPU (T4/A100)

## Repository Structure

```
deepfake-detection/
├── README.md
├── README_KR.md
├── .gitignore
├── notebooks/
│   ├── 00_data_preprocessing.ipynb         RetinaFace face extraction
│   ├── 01_xception_lstm.ipynb              Main: Xception + LSTM (video-level eval)
│   └── 02_video_swin_transformer.ipynb     Main: Video Swin Transformer + confusion matrix
├── _drafts/                                Earlier experiments preserved for transparency
│   ├── xception_lstm__early_with_viz.ipynb
│   ├── video_swin__with_training_history.ipynb   (11-epoch training history)
│   └── video_swin__initial_experiment.ipynb
└── checkpoints/
    └── README.md                            Best model weights (~82 MB) — stored externally
```

## Methods

### 1. Data Preprocessing (`00_data_preprocessing.ipynb`)
- Per-frame face detection with **RetinaFace** (`retina-face` package).
- Face crops aligned and saved as image sequences for downstream model input.
- Handles both single-face and multi-face frames.

### 2. Xception + LSTM (`01_xception_lstm.ipynb`)
- **Per-frame feature extractor:** ImageNet-pretrained Xception, 2048-d feature.
- **Temporal model:** LSTM over the frame-feature sequence per video clip.
- **Video-level decision:** aggregation across all face crops in the video
  (multi-face video-level evaluation included).

### 3. Video Swin Transformer (`02_video_swin_transformer.ipynb`)
- 3D shifted-window self-attention directly on the video volume
  (no separate per-frame extractor needed).
- Pretrained `swin3d_tiny` backbone, fine-tuned on the deepfake task.
- Evaluation: validation set + **confusion matrix visualization**.

## Stack

`Python` · `PyTorch` · `torchvision` · `retina-face` · `OpenCV` · `gdown` · `Google Colab`

## How to Run

```bash
# Open in Google Colab (recommended — preprocessing uses Colab Drive mount).
# 1. notebooks/00_data_preprocessing.ipynb   — extract face crops
# 2. notebooks/01_xception_lstm.ipynb        — or
#    notebooks/02_video_swin_transformer.ipynb
```

The notebooks download the (manipulated / original) datasets via `gdown` from Google Drive.
Datasets and trained weights are excluded from this repo (see `.gitignore`).

## Trained Weights

`best_model.pth` (~82 MB) is **not included** in this repo to keep size manageable.
Available on request — see [`checkpoints/README.md`](./checkpoints/README.md).

## Award

🥉 **3rd Place**, T-SUM Data Analysis Competition, Korea University Sejong (2023) · Team of 4

---

*Author: Sunjae Lee · B.S. Big Data Science @ Korea University Sejong*
*Part of [Computer-Vision](https://github.com/Vongole25/Computer-Vision)*
