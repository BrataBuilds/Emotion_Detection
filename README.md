# Emotion Recognition Model

This project implements a deep learning model for facial emotion recognition using TensorFlow and Modal. The model is capable of detecting seven different emotions from facial expressions in greyscale images.

## Overview

The emotion recognition system uses a Convolutional Neural Network (CNN) architecture to classify facial expressions into seven basic emotions. The model is trained on greyscale images of size 48x48 pixels and can be deployed either locally or using Modal's cloud infrastructure for GPU-accelerated training.

## Features

The project includes several key capabilities:

1. Deep Learning Model
   - Custom CNN architecture optimized for emotion recognition
   - Support for seven emotion categories
   - Dropout layers to prevent overfitting
   - Configurable learning rate and batch size

2. Data Processing
   - Automatic image preprocessing and normalization
   - Real-time data augmentation during training
   - Support for greyscale image processing
   - Efficient batch processing of training data

3. Training Infrastructure
   - GPU acceleration support through Modal
   - Configurable training parameters
   - Progress monitoring during training
   - Locally saved model weights and architecture.
## Prerequisites

Before running the project, ensure you have:

- Python 3.11 or higher
- A Modal account (for cloud training)
- Sufficient disk space for the dataset
- Git (for version control)


## Project Structure

```
├── train_local.py
|──train_on_modal
└── README.md
```

## Installation
1. Download the dataset from [here](https://www.kaggle.com/datasets/msambare/fer2013) and extract the files into the root folder.
2. Clone the repo:
```bash
git clone https://github.com/BrataBuilds/Emotion_Detection.git
```
3. Create a virtual environment (using uv preferred)
```bash
pip install uv
uv venv -p 3.11
```
4. Install required packages:
```bash
pip install -r requirements.txt
```
1. Set-up Modal:

Create an account on [modal.com](https://modal.com/docs/guide) and run the following command.
```bash
python -m modal setup
```
### Training the Model

1. Local Training:
```bash
python train.py
```

1. Modal Training (with GPU acceleration):
```bash
modal run train.py
```
