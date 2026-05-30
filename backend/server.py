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
from database import (
    init_db,
    create_test,
    list_all_records,
    list_patients_summary,
    get_patient_tests,
)

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


@app.route("/api/records", methods=["GET"])
def api_records():
    """All test records (flat list for dashboard / compatibility)."""
    ptype = request.args.get("type")
    if ptype and ptype not in ("normal", "emerg"):
        return jsonify({"error": "type must be normal or emerg"}), 400
    return jsonify({"records": list_all_records(ptype)})


@app.route("/api/patients", methods=["GET"])
def api_patients():
    """Patients with test counts and latest result."""
    ptype = request.args.get("type", "normal")
    if ptype not in ("normal", "emerg"):
        return jsonify({"error": "type must be normal or emerg"}), 400
    return jsonify({"patients": list_patients_summary(ptype)})


@app.route("/api/patients/<int:patient_id>/tests", methods=["GET"])
def api_patient_tests(patient_id):
    data = get_patient_tests(patient_id)
    if data is None:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify(data)


@app.route("/api/tests", methods=["POST"])
def api_save_test():
    """
    Save a completed test and link it to a patient (find-or-create by name + phone + type).
    """
    data = request.get_json(silent=True) or {}
    required = ("type", "name", "blood_group", "test_date", "test_time", "tested_by", "tested_by_role")
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    if data["type"] not in ("normal", "emerg"):
        return jsonify({"error": "type must be normal or emerg"}), 400

    try:
        saved = create_test(
            ptype=data["type"],
            name=data["name"],
            age=data.get("age", "—"),
            gender=data.get("gender", "—"),
            phone=data.get("phone", "—"),
            notes=data.get("notes"),
            blood_group=data["blood_group"],
            confidence=data.get("confidence"),
            test_date=data["test_date"],
            test_time=data["test_time"],
            tested_by=data["tested_by"],
            tested_by_role=data["tested_by_role"],
        )
        record = list_all_records()
        rec = next((r for r in record if r["id"] == saved["record_code"]), None)
        return jsonify({"ok": True, **saved, "record": rec}), 201
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print(f"🩸 HemaTrace backend server starting on {SERVER_HOST}:{SERVER_PORT} ...")
    init_db()
    load_model()   # pre-load model before accepting requests
    print(f"✅ Model ready. Listening ...")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
