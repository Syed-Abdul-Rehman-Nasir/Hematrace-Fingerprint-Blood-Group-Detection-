# backend/predict.py
"""
Inference with Test-Time Augmentation (TTA).
PyTorch checkpoint (hemtrace_best.pt). Same JSON contract for Flask / Electron.
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from config import CLASSES, INPUT_SIZE, MODEL_PATH, NUM_CLASSES
from preprocess import preprocess_single_image, augment_image
from model import HemaTraceNet


_model = None
_device = None


def _get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        device = _get_device()
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model = HemaTraceNet(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _model = model
        print(f"✅ Model loaded from {MODEL_PATH} ({device})")
    return _model


def predict_from_path(image_path, tta_steps=10):
    """
    Predict blood group from a fingerprint image file.

    Returns dict: predicted_class, confidence, all_scores, error
    """
    model = load_model()
    device = _get_device()
    H, W = INPUT_SIZE

    base = preprocess_single_image(image_path)
    if base is None:
        return {
            "error": f"Could not load image: {image_path}",
            "predicted_class": None,
            "confidence": 0,
            "all_scores": {},
        }

    versions = augment_image(base)[:tta_steps]
    batch_np = np.stack([v.reshape(H, W) for v in versions], axis=0)
    # (N, 1, H, W)
    x = torch.from_numpy(batch_np[:, np.newaxis, :, :].astype(np.float32)).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    idx = int(np.argmax(probs))
    conf = float(probs[idx]) * 100.0

    return {
        "predicted_class": CLASSES[idx],
        "confidence": round(conf, 2),
        "all_scores": {
            cls: round(float(probs[i]) * 100.0, 2)
            for i, cls in enumerate(CLASSES)
        },
        "error": None,
    }


def predict_from_bytes(image_bytes, tta_steps=10):
    import tempfile

    suffix = ".bmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(image_bytes)
        tmp_path = f.name
    try:
        return predict_from_path(tmp_path, tta_steps)
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    result = predict_from_path(sys.argv[1])
    if result["error"]:
        print(f"❌ {result['error']}")
    else:
        print(f"🩸 Blood Group : {result['predicted_class']}")
        print(f"📊 Confidence  : {result['confidence']:.1f}%")
        print(f"📈 All scores  : {result['all_scores']}")
