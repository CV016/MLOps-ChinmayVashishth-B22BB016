# B22BB016 — ML/DL Ops Midsem Exam

## Overview
Image classification on the **STL-10** dataset (10 classes) using **ResNet-18** built with PyTorch.

| Class | airplane | bird | car | cat | deer | dog | horse | monkey | ship | truck |
|---86.70----|----93.00------|---87.00---|---96.00--|--85.00---|---88.00---|--61.00---|---85.00----|---94.00-----|---90.00---|---88.00----|

---

## Repository Structure

### `B22BB016/` — Docker-based Training & Evaluation
| File | Purpose |
|------|---------|
| `train.py` | Trains ResNet-18 from scratch (3 epochs, Adam, LR=1e-3) on local `data/train/` and saves `trained_model.pth` |
| `evaluate.py` | Loads a saved model, computes accuracy, F1 score, classification report & single-image prediction on `data/test/` |
| `Dockerfile` | Python 3.10-slim container that installs deps and runs `train.py` |
| `requirements.txt` | torch, torchvision, numpy, scikit-learn, Pillow |

### `B22BB016-HuggingFace/` — Colab Notebook with HuggingFace & WandB
`B22BB016_ML_DL_Ops_Midsem_Exam.ipynb` — end-to-end pipeline:

1. **Data** — loads `Chiranjeev007/STL-10_Subset` from HuggingFace Datasets (5000 train / 500 val / 1000 test)
2. **Training** — ResNet-18 (ImageNet pretrained), 10 epochs, with data augmentation; best model saved via early stopping on val accuracy
3. **Logging** — training loss/accuracy curves logged to **Weights & Biases**
4. **Model Hub** — best model + plots pushed to HuggingFace repo `CV016/MDLOpsExam`
5. **Evaluation** — model pulled back from HuggingFace; test accuracy **86.70%**, confusion matrix, class-wise accuracy, and 20 sample predictions logged to WandB

---

## Quick Start

**Docker (B22BB016)**
```bash
cd B22BB016
docker build -t b22bb016-train .
docker run b22bb016-train
```

**Notebook (B22BB016-HuggingFace)**
Open `B22BB016_ML_DL_Ops_Midsem_Exam.ipynb` in Google Colab and run all cells.

---

## Results (Notebook)
| Metric | Value |
|--------|-------|
| Best Val Accuracy | 84.20% |
| Test Accuracy | 86.70% |

---

## Tools & Frameworks
PyTorch · torchvision · scikit-learn · Docker · HuggingFace Hub · Weights & Biases