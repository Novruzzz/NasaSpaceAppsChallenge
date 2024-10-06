import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import numpy as np
from obspy import read
import os
import librosa
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
checkpoint_file = "../Model/model.pt"  # Update this path to your model checkpoint
model.load_state_dict(torch.load(checkpoint_file, map_location=device))
model.to(device)
model.eval()

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
        S = librosa.feature.melspectrogram(y=data.astype(np.float128), sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Prepare the input for the model
        input_tensor = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).to(device)

        # Get the model prediction
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().tolist()

        return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5002)