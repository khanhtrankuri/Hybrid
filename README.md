# Hybrid Model Project

This project implements a Hybrid Backbone Mixture of Experts (MoE) model, including training, model merging (M2N2), multi-task fine-tuning, and evaluation.

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision datasets matplotlib numpy
```

## Project Structure

- `Hybrid/architech/archi.py`: Defines the model architectures (`VisionMoEViT`, `HybridBackboneMoE`, `HybridClassifier`).
- `Hybrid/main.py`: Script for training the base models.
- `Hybrid/merging_model.py`: Script for merging two checkpoints using the M2N2 algorithm.
- `Hybrid/finetune.py`: Script for multi-task fine-tuning (CIFAR-10 + CIFAR-100) of the merged model.
- `Hybrid/evaluate.py`: Script for evaluating the model and visualizing predictions.

## Usage Instructions

### 1. Train Base Models

Train the model to generate checkpoints. You may need to run this twice (or modify configurations) to generate different checkpoints for merging (e.g., `cifar100_hybrid_moe.pth` and `cifar10_hybrid_moe.pth`).

```bash
python Hybrid/main.py
```

*Note: Check `Hybrid/main.py` to adjust hyperparameters or dataset (CIFAR-10 vs CIFAR-100).*

### 2. Merge Models (M2N2)

Merge two trained checkpoints into a single model.

```bash
python Hybrid/merging_model.py
```

This will read `checkpoints/cifar100_hybrid_moe.pth` and `checkpoints/cifar10_hybrid_moe.pth` (make sure these exist) and save the merged model to `checkpoints/cifar_hybrid_moe_merged.pth`.

### 3. Multi-task Fine-tuning

Fine-tune the merged backbone simultaneously on CIFAR-10 and CIFAR-100.

```bash
python Hybrid/finetune.py
```

This loads `checkpoints/cifar_hybrid_moe_merged.pth` and saves the fine-tuned model to `checkpoints/cifar_multitask_finetuned.pth`.

### 4. Evaluation and Visualization

Evaluate the fine-tuned model on CIFAR-10 and visualize predictions.

```bash
python Hybrid/evaluate.py
```

This will:
- Load `checkpoints/cifar_multitask_finetuned.pth`.
- Calculate accuracy on the CIFAR-10 test set.
- Generate a visualization image `evaluation_results.png` showing predicted vs. true labels.