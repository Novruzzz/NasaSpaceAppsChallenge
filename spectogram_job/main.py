from flask import Flask, request, jsonify
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from obspy import read
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
        file_path = os.path.join("/tmp/spectrogram", file.filename)
        file.save(file_path)

        # Read the mseed file
        st = read(file_path)
        print(st)
        tr = st[0]  # Assuming we are working with the first trace
        print(tr)
        data = tr.data
        print(data)
        sr = tr.stats.sampling_rate
        print(sr)

        # Generate a mel spectrogram
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Save the spectrogram as an image
        spectrogram_path = file_path.replace(".mseed", ".png")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='plasma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(spectrogram_path)
        plt.close()

        # Send the spectrogram image to the blob server
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