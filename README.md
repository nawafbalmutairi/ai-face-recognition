# AI Face Recognition & Gender Classification

> Azure ML Designer pipeline benchmarking **DenseNet** against **ResNet** for gender classification on facial images, evaluated in both training and real-time inference modes. The winning configuration (DenseNet on real-time inference) achieves **86.67%** accuracy across all three metrics on an unseen 30-image evaluation set.

[![Status](https://img.shields.io/badge/status-completed-success)]()
[![Accuracy](https://img.shields.io/badge/best%20accuracy-86.67%25-e64d2e)]()
[![Configurations](https://img.shields.io/badge/configurations-4-blue)]()
[![Azure ML](https://img.shields.io/badge/Azure%20ML-Designer-0078d4)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## TL;DR

| | |
|---|---|
| **Task** | Binary classification of facial images into Male / Female. |
| **Data** | 130 images total — Kaggle `gender_images` (100, 50/50) + self-curated `Nawaf_data_inference` (30, 15/15). |
| **Models** | DenseNet vs ResNet — two architectures, two evaluation modes. |
| **Configurations** | 4: {DenseNet, ResNet} × {Training pipeline, Real-time inference}. |
| **Winner** | **DenseNet on real-time inference — 0.8667** across precision, recall, and accuracy. |

---

## Results · The four configurations

The whole experiment is structured around a single controlled comparison: same preprocessing, two architectures, two evaluation modes. That isolates **architecture as the only variable**.

| Configuration | Precision | Recall | Accuracy | Notes |
|---|---:|---:|---:|---|
| DenseNet · Training | 0.8000 | 0.8000 | 0.8000 | Balanced baseline |
| ResNet · Training | 0.7376 | 0.7333 | 0.7333 | ~6.7 points behind DenseNet |
| **DenseNet · Real-time inference** | **0.8667** | **0.8667** | **0.8667** | ★ Best overall |
| ResNet · Real-time inference | 0.8000 | 0.8000 | 0.8000 | ~7 points behind DenseNet |

**Three things to notice:**

1. **DenseNet wins both modes.** Its dense connectivity preserves feature information through the network better than ResNet's residual connections on this small dataset.
2. **DenseNet *improves* on the unseen inference set** (0.80 → 0.87). The 30-image inference set isn't harder than the training set's holdout — DenseNet generalises well to my self-curated photos.
3. **Errors are balanced across classes.** Precision, recall, and accuracy all land at the same number — meaning male misclassifications and female misclassifications happen at equal rates, not skewed.

---

## The Azure ML Designer pipeline

### Training pipeline (5 stages)

```
Dataset (gender_images)  →  Convert & Split  →  Transform  →  Train (PyTorch)  →  Score & Evaluate
   ⏵ 100 imgs                ⏵ 70/30 splits     ⏵ identical     ⏵ GPU compute       ★ metrics out
```

| Stage | Component | Notes |
|---|---|---|
| 1 | `Dataset` | Kaggle gender_images (100 photos, 50M / 50F) |
| 2 | `Convert to Image Directory` → `Split Image Directory` | First 70/30 train/test, then 70/30 train/val |
| 3 | `Init Image Transformation` + 3× `Apply Image Transformation` | Same pipeline for train, val, test |
| 4 | `DenseNet` or `ResNet` → `Train PyTorch Model` | GPU compute target |
| 5 | `Score Image Model` → `Evaluate Model` | Macro & micro precision, recall, accuracy |

### Real-time inference pipeline (4 stages)

After both models are trained, a second pipeline applies them to a new 30-image dataset I curated myself. This is the **generalisation test** — can the model recognise faces it has never seen, that weren't sampled from the same source as training?

```
New dataset                →  Convert & Transform  →  Score (trained model)  →  Evaluate
   ⏵ Nawaf_data_inference     ⏵ Azure              ⏵ DenseNet / ResNet         ★ 86.67%
   ⏵ 30 photos (15M / 15F)
```

---

## Why this design

A common mistake in image-classification benchmarks is **changing more than one thing per experiment** — different preprocessing, different optimiser, different architecture, all at once. The result can't isolate which change drove the metric movement.

This project deliberately holds everything constant *except* the architecture: same Convert to Image Directory, same 70/30 splits, same Init + Apply Image Transformation chain, same PyTorch training step. The only difference is whether the network is DenseNet or ResNet.

**That's what makes the 6.7-point gap meaningful.**

---

## Repository structure

```
.
├── README.md                                  ← you are here
├── LICENSE                                    ← MIT
├── .gitignore                                 ← Python + Azure ML ignores
├── pipelines/
│   ├── training_pipeline.json                 ← exported Azure ML Designer JSON
│   ├── inference_pipeline.json                ← exported inference pipeline
│   └── pipeline_diagrams/
│       ├── training_densenet.png              ← Azure visual export
│       └── training_resnet.png
├── data/
│   ├── gender_images/                         ← Kaggle source (link below)
│   └── Nawaf_data_inference/                  ← 30 self-curated photos
├── results/
│   ├── densenet_training_metrics.json         ← 0.80 / 0.80 / 0.80
│   ├── resnet_training_metrics.json           ← 0.74 / 0.73 / 0.73
│   ├── densenet_inference_metrics.json        ← 0.87 / 0.87 / 0.87 ★
│   └── resnet_inference_metrics.json          ← 0.80 / 0.80 / 0.80
├── notebooks/
│   └── compare_models.ipynb                   ← side-by-side comparison
└── docs/
    └── final_report.pdf                       ← full submitted report
```

> **Data licensing note.** The 100-image Kaggle `gender_images` set is not redistributed here; download from Kaggle and place in `data/gender_images/`. The 30-image `Nawaf_data_inference` set is my own and not included for privacy reasons.

---

## Reproduce the result

### Prerequisites

- An [Azure ML workspace](https://learn.microsoft.com/en-us/azure/machine-learning/) (free tier works)
- Azure ML Designer enabled in your workspace
- ~1 GB free in your workspace blob storage

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/nawafbalmutairi/ai-face-recognition.git
cd ai-face-recognition

# 2. Upload datasets to your Azure ML workspace
# (via Azure ML Studio UI → Data → Create dataset → From local files)

# 3. Import the pipelines
# In Azure ML Designer → New pipeline → Import from JSON
# Use pipelines/training_pipeline.json
```

Run the training pipeline against `gender_images` first to produce the trained model. Then run the inference pipeline against `Nawaf_data_inference` (or your own 15/15 male/female set) to reproduce the 0.8667 result.

Expected runtime on default Azure GPU compute (`STANDARD_NC6`): **~12 minutes** per training run.

---

## Tech stack

**Cloud:** Microsoft Azure ML · Azure ML Designer · Azure ML Studio
**ML framework:** PyTorch (via Train PyTorch Model component)
**Architectures:** DenseNet · ResNet
**Compute:** Azure GPU compute (`STANDARD_NC6` recommended)
**Evaluation:** macro / micro precision, recall, accuracy

---

## What's interesting about this dataset

The 100-image Kaggle set is tiny by computer-vision standards (ImageNet has 14M images, the equivalent gender classification work in literature typically uses CelebA at 200K+). On such small data:

- **Transfer learning matters more than architecture choice.** Both DenseNet and ResNet here are pretrained on ImageNet — the comparison is really about which transferred features generalise better to a small new domain.
- **Generalisation is a sharper test than training accuracy.** DenseNet's improvement from 0.80 (training) to 0.87 (real-time inference) on an unseen 30-image set is unusual — it suggests the training-set holdout was actually *harder* than the new set, which is the opposite of the typical pattern.
- **30 images is too few to claim a definitive winner.** A proper benchmark would use 1000+ unseen images. The 7-point gap is real, but the confidence interval is wide.

---

## Author

**Nawaf Almutairi** — BSc Computer Science, Northumbria University · Class of 2026

[Portfolio](https://nawafbalmutairi.github.io) · [LinkedIn](https://linkedin.com/in/nawaf-almutairi-907766290/) · [Email](mailto:NawafBAlmutairi@outlook.sa)

---

## License

[MIT](LICENSE) — see file for full text. The 30 photos in `Nawaf_data_inference/` are not included in this repository for privacy reasons.
