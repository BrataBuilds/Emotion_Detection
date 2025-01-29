"""
README 
This program allows you to see the performance of the model in real time by taking inputs of your face cam.
Each frame is stored, transformed to our model's requirements, and its output showcased beside.
OpenCv is used to load this data and its pre-trained Haar Cascade model for detecting faces in images.
"""
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
from PIL import Image
import numpy as np
# Re-defining the CNN Architecture
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten_size = 128 * 6 * 6  
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc1_relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)  
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)        
        x = self.conv2(x)        
        x = self.pool1(x)
        x = self.dropout1(x)        
        x = self.conv3(x)        
        x = self.pool2(x)        
        x = self.conv4(x)        
        x = self.pool3(x)
        x = self.dropout2(x)        
        # Flatten the output
        x = x.view(x.size(0), -1)        
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

# Using the Haar Cascade model for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the pretrained emotion detection model on the CPU to locally running the model,
# Main reason for this is privacy concern.
model = EmotionCNN()
model.load_state_dict(torch.load('models/final_model(2).pth', map_location=torch.device('cpu')))
model.eval()  


emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((48,48)),  # Resizing the image to match the original model training data
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale as our as per the model training data
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  
])

def detect_faces_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input_face = transform(pil_face).unsqueeze(0) 
            input_face = Variable(input_face)

            # Predict emotion
            with torch.no_grad():  # Disable gradient calculation as we are not training the data.
                output = model(input_face)
                _, predicted_class = torch.max(output, 1)
                predicted_emotion = emotion_labels[predicted_class.item()]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam - Emotion Detection", frame)

        # Exit the app by pressing Q.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_faces_webcam()