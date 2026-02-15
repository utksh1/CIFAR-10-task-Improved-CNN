# CIFAR-10 Small-Data Experiments

This repository contains a technical challenge to train and improve a CNN on a small subset (10,000 samples) of the CIFAR-10 dataset.

## Highlights
- **Baseline Accuracy**: 61.66% (3-layer CNN)
- **Improved Accuracy**: 69.31% (Added Batch Norm + Data Augmentation)
- **Key finding**: Batch Normalization and Data Augmentation are critical when working with limited training data to prevent overfitting and stabilize gradients.

## Contents
- `cifar10_cnn_experiment.ipynb`: A step-by-step notebook with results and technical reasoning.
- `cifar_training.py`: The core script used for experiments.
- `plots/`: Contains the loss and accuracy comparison plots.

## How to run locally
1. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```
2. Run the script:
   ```bash
   python cifar_training.py
   ```
   Or open the `experiment_results.ipynb` in Jupyter/Colab.
