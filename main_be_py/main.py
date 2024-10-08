import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

spectrogram_server_url = "http://localhost:5001/upload"
classifier_server_url = "http://localhost:5002/upload"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join("/tmp/backend", file.filename)
        file.save(file_path)

        with open(file_path, 'rb') as img_file:
            response = requests.post(spectrogram_server_url, files={'file': img_file})
            if response.status_code == 200:
                spectrogram_url = response.json()["url"]
            else:
                return jsonify({"error": "Failed to upload file to spectrogram server"}), 500

        with open(file_path, 'rb') as img_file:
            response = requests.post(classifier_server_url, files={'file': img_file})
            if response.status_code == 200:
                prediction = response.json()["prediction"]
            else:
                return jsonify({"error": "Failed to upload file to classifier server"}), 500

        return jsonify({"spectrogram_url": spectrogram_url, "prediction": prediction})
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)