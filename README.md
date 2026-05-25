<!--
  Vongole25/Computer-Vision/README.md
  → CV 관련 프로젝트 + 아키텍처 구현 repo
-->

# Computer Vision — Implementations & Projects

Personal repository for **Computer Vision implementations** and **competition projects**.
Code reproductions of classic / modern architectures, plus award-winning competition work.

> Author: **Sunjae Lee** · B.S. Big Data Science @ Korea University Sejong · [Blog](https://it-study-2002.tistory.com/)

---

## 🏆 Competition Projects

### Defective Coffee Bean Classification — 🥈 2nd Place
*T-SUM Data Analysis Competition, Korea University Sejong (2023) · Team of 4*

- **Task:** Multi-class classification of defective vs normal coffee beans from images.
- **Approach:**
  - Backbone: **EfficientNet-B0** with ImageNet-pretrained weights.
  - Data augmentation via Albumentations (rotation, color jitter, cutout).
  - Class-balanced sampling for imbalanced defect categories.
- **Result:** 2nd place out of intramural participants.
- 📁 `coffee-bean-classification/` (to be added)

### Deepfake Detection — 🥉 3rd Place
*T-SUM Data Analysis Competition, Korea University Sejong (2023) · Team of 4*

- **Task:** Binary classification of authentic vs deepfake video frames.
- **Approach:**
  - Frame extraction → CNN backbone feature extraction.
  - Investigated simple temporal aggregation (mean/max pooling over frames).
- **Result:** 3rd place.
- 📁 `deepfake-detection/` (to be added)

---

## 🧪 Architecture Reproductions

Reading & reproducing classic / modern CV papers as study notes.

- [ ] ResNet (He et al., 2015) — from scratch on CIFAR-10
- [ ] EfficientNet (Tan & Le, 2019)
- [ ] Vision Transformer (Dosovitskiy et al., 2021)
- [ ] CLIP (Radford et al., 2021) — exploring vision-language alignment
- [ ] DINO / DINOv2 — self-supervised representation

Each implementation lives in its own subfolder with:
- A short **README** explaining the paper takeaways.
- **Notebook** with training/eval curves.
- **Reference** paper link.

---

## 🛠 Stack

`Python 3.10+` · `PyTorch 2.x` · `torchvision` · `Albumentations` · `OpenCV` · `Jupyter`

---

## 📌 Related

- **[University-Projects](https://github.com/Vongole25/University-Projects)** — other coursework
- **[NLP](https://github.com/Vongole25/NLP)** — NLP study notes (coming soon: multimodal experiments)

📌 Currently preparing for **NLP / multimodal graduate school admission (2027 Spring)** — see [profile](https://github.com/Vongole25).
