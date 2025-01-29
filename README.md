# Emotion Recognition Model

This project implements a deep learning model for facial emotion recognition using TensorFlow and Modal. The model is capable of detecting seven different emotions from facial expressions in greyscale images.

## Overview of the Model

The emotion recognition system uses a Convolutional Neural Network (CNN) architecture to classify facial expressions into seven basic emotions. The model is trained on greyscale images of size 48x48 pixels and can be deployed using Modal's cloud infrastructure for high-end GPU-accelerated training.

Using `torch` we define a class `EmotionCNN` extends `torch.nn.Module` and defines a CNN that processes grayscale images (1-channel input) and classifies them into 7 different emotion classes. It consists of:

- 4 convolutional layers (feature extraction)
- 3 max-pooling layers (downsampling)
- Dropout layers (regularization)
- 2 fully connected layers (classification)
- Softmax activation (probability output)

A visual of how data flows through the network:
```py
def forward(self, x):
    x = self.conv1(x)        # Applies the first convolution
    x = self.conv2(x)        # Applies the second convolution
    x = self.pool1(x)        # Pooling
    x = self.dropout1(x)     # Dropout
    x = self.conv3(x)        # Applies the third convolution
    x = self.pool2(x)        # Pooling
    x = self.conv4(x)        # Applies the fourth convolution
    x = self.pool3(x)        # Pooling
    x = self.dropout2(x)     # Dropout

    # Flatten before passing into fully connected layers
    x = x.view(x.size(0), -1)  
    x = self.fc1(x)         # Fully connected layer
    x = self.fc1_relu(x)    # ReLU activation
    x = self.dropout3(x)    # Dropout
    x = self.fc2(x)         # Output layer
    x = self.softmax(x)     # Softmax activation
```
## CNN Output Shape Transformation
| Layer          | Output Shape (assuming input is 48x48) |
| -------------- | -------------------------------------- |
| Input Image    | (1, 48, 48)                            |
| Conv1 → ReLU   | (32, 48, 48)                           |
| Conv2 → ReLU   | (64, 48, 48)                           |
| MaxPool1 (2x2) | (64, 24, 24)                           |
| Conv3 → ReLU   | (128, 24, 24)                          |
| MaxPool2 (2x2) | (128, 12, 12)                          |
| Conv4 → ReLU   | (128, 12, 12)                          |
| MaxPool3 (2x2) | (128, 6, 6)                            |
| Flatten        | 4608 (128 * 6 * 6)                     |
| FC1 → ReLU     | 1024                                   |
| FC2 → Softmax  | 7                                      |
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

1. Modal Training (with GPU acceleration):
```bash
modal run train.py
```
2. Download the model weights locally and save it a directory.
3. Run the emotion_detection_cam.py to study the model's performance using your own face.
4. Run the song_recommend.py to get your song recommendation using the model's predicted output.
