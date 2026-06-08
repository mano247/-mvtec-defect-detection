# MVTec Defect Detection

Industrial defect detection using deep learning on MVTec AD dataset.

## Problem
Binary classification of industrial product images as good or defective across 15 product categories.

## Dataset
- MVTec AD dataset — 15 categories, ~4500 images
- 3629 good images, 849 defect images

## Models
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|---------|-----------|--------|-----|---------|
| CNN (from scratch) | 65.30% | 62.59% | 63.81% | 63.20% | 0.728 |
| ResNet18 (transfer learning) | 89.38% | 98.77% | 78.24% | 87.31% | 0.966 |
| ResNet50 (transfer learning) | 88.58% | 97.83% | 77.26% | 86.34% | 0.954 |
| Autoencoder | 59.25% | 75.49% | 18.83% | 30.14% | 0.490 |

## Key Findings
- ResNet18 outperforms all models with 89.38% accuracy and AUC-ROC of 0.966
- Larger model (ResNet50) doesn't always mean better results on small datasets
- Autoencoder generalizes too well and fails to distinguish defects (Recall only 18.83%)
- Textural categories (carpet, leather) are harder to classify than structural ones
- Per-category ResNet18 models achieved median accuracy of 83.33% vs 89.38% for single model

## Tech Stack
- Python, PyTorch, torchvision
- scikit-learn, seaborn, matplotlib
- Grad-CAM for model interpretability