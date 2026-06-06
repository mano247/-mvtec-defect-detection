# MVTec Defect Detection

Industrial defect detection using deep learning on MVTec AD dataset.

## Problem
Binary classification of industrial product images as good or defective across 15 product categories.

## Dataset
- MVTec AD dataset — 15 categories, ~4500 images
- 3629 good images, 849 defect images

## Models
| Model | Accuracy | AUC-ROC |
|-------|---------|---------|
| CNN (from scratch) | 67.69% | 0.728 |
| ResNet18 (transfer learning) | 88.13% | 0.966 |
| ResNet50 (transfer learning) | 87.67% | 0.954 |
| Autoencoder | 54.45% | 0.490 |

## Key Findings
- ResNet18 outperforms all models with 88.13% accuracy
- Larger model (ResNet50) doesn't always mean better results on small datasets
- Autoencoder generalizes too well and fails to distinguish defects
- Textural categories (carpet, leather) are harder to classify than structural ones

## Tech Stack
- Python, PyTorch, torchvision
- scikit-learn, seaborn, matplotlib
- Grad-CAM for model interpretability