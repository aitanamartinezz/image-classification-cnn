# Image Classification with Convolutional Neural Networks (CNN)

This repository contains the implementation of deep learning models based on
Convolutional Neural Networks (CNNs) for image classification tasks. The project
explores CNN architectures and training strategies using Python and popular
deep learning libraries.

---

## Project Overview

Convolutional Neural Networks are a fundamental architecture in computer vision,
particularly effective for image classification problems. This project focuses on
building, training, and evaluating CNN models to automatically classify images
based on learned visual features.

The implementation is designed to provide a clear and practical understanding of
CNN-based deep learning workflows, from data preprocessing to model evaluation.

---

## Repository Structure

 - P1_CNNS_AITANAMartinez.ipynb # Main notebook with full implementation
 - README.md # Project documentation


---

## Technologies

- Python
- Deep Learning
- Convolutional Neural Networks (CNN)
- Jupyter Notebook

---

## Data Preprocessing

The preprocessing pipeline includes:
- Image loading and resizing
- Normalization of pixel values
- Dataset splitting into training and validation sets
- Batch generation for efficient training

---

## Model Architecture

The CNN architecture is composed of:
- Convolutional layers for feature extraction
- Activation functions to introduce non-linearity
- Pooling layers for spatial downsampling
- Fully connected layers for classification
- Softmax output layer for class prediction

Regularization techniques are applied to improve generalization and reduce overfitting.

---

## Training Configuration

- Loss function suitable for multi-class classification
- Optimizer for gradient-based learning
- Training performed over multiple epochs
- Performance monitored using validation metrics

---

## Results

The trained CNN models demonstrate effective learning of visual patterns and achieve
strong classification performance on the evaluation dataset.

Detailed results, training curves, and model behavior can be found directly in the
Jupyter notebook.

---

## Usage

To run the project:

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook P1_CNNS_AITANAMartinez.ipynb
2. Execute the notebook cells sequentially to reproduce the full pipeline, including
data preprocessing, model training, and evaluation.
