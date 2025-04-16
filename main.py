from flask import Flask, request, jsonify
import librosa
import numpy as np
from prediction_method import predict_model
from tensorflow import keras
from keras import models

app = Flask(__name__)
model = models.load_model("tajwid_model.h5")
labels = ['idgham', 'idzhar', 'ikhfa', 'iqlab']

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    y, sr = librosa.load(file, sr=22050)

    results = predict_model(y, sr, model, labels)
    return jsonify({"results": results})

@app.route("/")
def index():
    return jsonify({"message": "Welcome to tajwid ai"})

# if __name__ == "__main__":
#     app.run()