# backend/server.py
"""
Lightweight Flask server.
Electron's main.js spawns this on app launch.
Exposes one endpoint: POST /predict

Request:  multipart/form-data  with field "image" (the .bmp file)
Response: JSON { predicted_class, confidence, all_scores, error }

CORS is restricted to 127.0.0.1 (localhost only — never exposed to internet).
"""

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from config  import SERVER_HOST, SERVER_PORT
from predict import predict_from_bytes, load_model

app = Flask(__name__)
CORS(app, origins=[f"http://{SERVER_HOST}:{SERVER_PORT}",
                   "null",           # Electron file:// pages appear as "null" origin
                   "file://"])


@app.route("/health", methods=["GET"])
def health():
    """Electron polls this to know when the server is ready."""
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a fingerprint image file and returns blood group prediction.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request",
                        "predicted_class": None,
                        "confidence": 0,
                        "all_scores": {}}), 400

    file_bytes = request.files["image"].read()
    if len(file_bytes) == 0:
        return jsonify({"error": "Empty image file",
                        "predicted_class": None,
                        "confidence": 0,
                        "all_scores": {}}), 400

    result = predict_from_bytes(file_bytes, tta_steps=10)
    status  = 500 if result.get("error") else 200
    return jsonify(result), status


if __name__ == "__main__":
    print(f"🩸 HemaTrace backend server starting on {SERVER_HOST}:{SERVER_PORT} ...")
    load_model()   # pre-load model before accepting requests
    print(f"✅ Model ready. Listening ...")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
