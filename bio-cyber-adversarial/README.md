# Bio-Cybersecurity Adversarial Benchmark

## Overview
This repository contains the first experimental dataset and baseline pipeline for a Deep Learning / Bio-Cybersecurity research project. The goal of this project is to study adversarial attacks on sequence-based machine-learning models.

**IMPORTANT:** This is a **synthetic computational benchmark**, not a model of a real biological system. Do NOT use real biological sequences or real pathogenic/functional motifs. The purpose is to study adversarial behavior safely in a controlled environment.

## What this experiment is testing
This experiment tests whether a simple 1D Convolutional Neural Network (CNN) can successfully learn to classify synthetic DNA-like sequences based on implanted synthetic motifs. We establish a baseline classifier and dataset that will be used in future steps to train an adversarial attack agent.

## Dataset
- **Alphabet**: A, C, G, T
- **Sequence length**: 50
- **Total samples**: 20,000
- **Classes**: 0 and 1
- **Motifs**: 
  - Class 0: `ACGTACGT`
  - Class 1: `TGCATGCA`

The sequences are composed of random background characters with the synthetic motif implanted at a random position. 

### Generation
The dataset generation is fully deterministic and uses a stratified train/validation/test split.
Run the generator:
```bash
python src/data_generator.py
```
This saves a `dataset.csv` in `data/raw/`.

## Encoding and Model
Sequences are one-hot encoded into a PyTorch tensor of shape `(4, sequence_length)` (A=[1,0,0,0], C=[0,1,0,0], etc.).
The baseline model is a simple 1D CNN with max pooling and fully connected layers.

## Training
To train the baseline model:
```bash
python src/train.py
```
This script tracks training/validation loss and accuracy, and saves the best model checkpoint to `models/best_model.pth`.

## Evaluation
To evaluate the trained model on the test set:
```bash
python src/evaluate.py
```
This will report Accuracy, Precision, Recall, F1 score, and the Confusion Matrix. 
It also includes an **interpretability sanity check** to verify that the model's predictions actually depend on the implanted motif, rather than accidental dataset artifacts.
