import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from obspy import read
import os
import librosa

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
class CNNQuakeDetector(nn.Module):
    def __init__(self, in_shape=(1, 1, 5, 129), out_channels=64, n_classes=1) -> None:
        super(CNNQuakeDetector, self).__init__()

        B, C, W, H = in_shape

        self.CNN = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2)
            )

        self.Classifier = nn.Linear(out_channels * (W // 4) * (H // 4), n_classes)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, X):
        out = self.CNN(X)
        out = self.Classifier(out)
        out = self.Sigmoid(out)
        return out

model = CNNQuakeDetector(n_classes=1)
checkpoint_file = "/content/drive/MyDrive/NSAC/model.pt"
model.load_state_dict(torch.load(checkpoint_file, map_location=device))
model.to(device)
model.eval()

# Define the dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# Load the training data
data = torch.load("/data/NSAC/data.pt", weights_only=False)
targets = torch.load("/data/NSAC/targets.pt", weights_only=False)
dataset = Dataset(data, targets)

# Define the route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # Save the uploaded file
        file_path = os.path.join("/tmp", file.filename)
        file.save(file_path)

        # Read the mseed file
        st = read(file_path)
        tr = st[0]  # Assuming we are working with the first trace
        data = tr.data
        sr = tr.stats.sampling_rate

        # Generate a mel spectrogram
        S = librosa.feature.melspectrogram(y=data.astype(float), sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Prepare the input for the model
        input_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).to(device)

        # Get the model prediction
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().tolist()

        return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)