from flask import Flask, request, jsonify
import os
from obspy import read
from flask_cors import CORS
import random as rand

app = Flask(__name__)
CORS(app)

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

        # Read the mseed file (mock processing)
        st = read(file_path)
        tr = st[0]  # Assuming we are working with the first trace
        data = tr.data
        sr = tr.stats.sampling_rate

        recall = rand.uniform(0.985, 0.9999999)

        # Mock prediction response
        prediction = {
            "loss": 0.043321698904037476,
            "quake": recall > 0.5,
            "recall": recall,
            "precision": recall,
            "f1": recall,
        }

        return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5002)