# Understanding Deep Learning: Image Classification for Myringosclerosis Diagnosis

**Author:** Mayra Fuentes Pineda
**Context:** Coursework — Master's in Healthcare Big Data

## Overview

This project explores the fundamentals of deep learning and Convolutional
Neural Networks (CNNs) by building and comparing two image classification
models that distinguish **myringosclerosis** (calcified deposits on the
tympanic membrane) from healthy eardrums using otoscopic images.

It served as a hands-on precursor to deeper work on automated otoscopy
classification, focused specifically on understanding how CNNs learn,
rather than on achieving a final clinical-grade model.

## Dataset

**Otoscopic Image Dataset** (Kaggle) — 3,000 JPEG otoscopy images across
5 clinical categories. Only the **Myringosclerosis** and **Normal** classes
were used (1,200 images total), split 70% train / 15% validation / 15% test.
Images were resized from 500×500 to 100×100 px to reduce computational cost
and overfitting risk.

## Models

- **Model 1 — CNN from scratch:** A sequential CNN (3 convolutional blocks +
  dense classifier, ~5.4M parameters) built from the ground up to understand
  core CNN mechanics — convolution, pooling, and feature map progression.
- **Model 2 — ResNet50 (transfer learning):** Pretrained on ImageNet, with
  all convolutional layers frozen and new dense layers trained on top as a
  classifier head (~1M trainable parameters).

Both models were trained with early stopping and model checkpointing,
using the Adam optimizer and binary cross-entropy loss.

## Results

| Model | F1-Score | Validation Accuracy | Notes |
|---|---|---|---|
| CNN Scratch | 0.89 | 93.9% | Took longer to converge (26 epochs); more prone to misclassifying healthy eardrums as pathological |
| ResNet50 (transfer learning) | 0.98 | 98.3% | Converged in 9 epochs; better class balance and generalization |

ResNet50 clearly outperformed the from-scratch model in both accuracy and
training efficiency — as expected, given its pretrained feature extraction
capability — while the scratch model remained valuable as a learning
exercise in CNN architecture and hyperparameter behavior.

## Software

**Python**, **TensorFlow/Keras**, run on **Google Colab** (GPU runtime).

## Repository contents

- `docs/MScThesis_fuentes_presentation.pptx` — Full write-up (in Spanish)
- `docs/MScThesis_fuentes.pdf` — Full write-up (in Spanish)
- `notebooks/` — Model development and training notebooks

## Tech stack

`Python` · `TensorFlow` · `Keras` · `Google Colab`
