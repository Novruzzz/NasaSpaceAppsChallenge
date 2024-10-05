from flask import Flask, request, jsonify
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

app = Flask(__name__)

BLOB_SERVER_URL = "http://localhost:8080/blob"

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

        df = pd.read_csv(file_path)

        if not {'time_abs(%Y-%m-%dT%H:%M:%S.%f)', 'time_rel(sec)', 'velocity(m/s)'}.issubset(df.columns):
            return jsonify({"error": "CSV file must contain 'time_abs(%Y-%m-%dT%H:%M:%S.%f)', 'time_rel(sec)', 'velocity(m/s)' columns"}), 400

        speed = df['velocity(m/s)'].values
        sr = 1 / np.mean(np.diff(df['time_rel(sec)'].values))

        # S = librosa.feature.melspectrogram(y=speed, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S = librosa.feature.mfcc(y=speed, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)

        spectrogram_path = file_path.replace(".csv", ".png")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='plasma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(spectrogram_path)
        plt.close()

        with open(spectrogram_path, 'rb') as img_file:
            response = requests.put(BLOB_SERVER_URL, files={'file': img_file})
            if response.status_code == 200:
                key = response.text
                file_url = f"{BLOB_SERVER_URL}/{key}"
                return jsonify({"url": file_url})
            else:
                return jsonify({"error": "Failed to upload file to blob server"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)