'''
README
Original idea was to use the spotify / youtube api to recommend songs however I don't have experience with requests and didn't have time.
Current implementation simply detects the emotion and then searches the song from the list below on the browser through google
TODO
1. Implement Spotify or Youtube api to search songs there.

'''
import random
import webbrowser
import time
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
from PIL import Image

# Songs based on type of emotion predicted by the model.
emotion_to_songs = {
    "Angry": ["Breaking the Habit - Linkin Park", "Stronger - Kanye West", "Killing in the Name - Rage Against the Machine"],
    "Disgust": ["I Don't Care - Ed Sheeran & Justin Bieber", "Take a Bow - Rihanna", "Rolling in the Deep - Adele"],
    "Fear": ["Disturbia - Rihanna", "Boulevard of Broken Dreams - Green Day", "Thriller - Michael Jackson"],
    "Happy": ["Happy - Pharrell Williams", "Can't Stop the Feeling! - Justin Timberlake", "Walking on Sunshine - Katrina and the Waves"],
    "Sad": ["Someone Like You - Adele", "Fix You - Coldplay", "The Scientist - Coldplay"],
    "Surprise": ["Wake Me Up - Avicii", "Viva La Vida - Coldplay", "Bohemian Rhapsody - Queen"],
    "Neutral": ["Let It Be - The Beatles", "Chasing Cars - Snow Patrol", "Imagine - John Lennon"]
}

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

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
        x = x.view(x.size(0), -1)        
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = EmotionCNN()
model.load_state_dict(torch.load('model/final_model(1).pth', map_location=torch.device('cpu')))
model.eval()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def track_emotion():
    cap = cv2.VideoCapture(0)
    detected_emotions = []
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input_face = transform(pil_face).unsqueeze(0)
            input_face = Variable(input_face)
            with torch.no_grad():
                output = model(input_face)
                _, predicted_class = torch.max(output, 1)
                detected_emotions.append(emotion_labels[predicted_class.item()])
    cap.release()
    return detected_emotions

# Doing a quick google search to the recommended song.
def recommend_song(detected_emotions):
    if not detected_emotions:
        return "No emotion detected. Please try again."
    emotion_counts = {emotion: detected_emotions.count(emotion) for emotion in set(detected_emotions)}
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    recommended_song = random.choice(emotion_to_songs[dominant_emotion])
    search_query = f"https://www.google.com/search?q={recommended_song.replace(' ', '+')}+song"
    webbrowser.open(search_query)
    return f"Detected emotion: {dominant_emotion}\nRecommended song: {recommended_song}\nOpening browser for search..."

detected_emotions = track_emotion()
print(recommend_song(detected_emotions))
