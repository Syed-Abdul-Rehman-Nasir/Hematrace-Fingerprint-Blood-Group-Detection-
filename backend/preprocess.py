# backend/preprocess.py
"""
Preprocessing pipeline for fingerprint images.
Key fixes vs old notebook:
  - Input resized to 224×224 (was 128×128 — too small for ridge detail)
  - Gabor filter bank added (explicitly extracts ridge orientation patterns)
  - Horizontal flip REMOVED from augmentation (mirror fingerprints don't exist)
  - Augmentation limited to rotations ±10°, small brightness shifts, zoom ±5%
  - All augmentation confined to spatial transforms that fingerprints undergo
    naturally (slight rotation, pressure variation via brightness).
"""

import cv2
import numpy as np
import random
from config import INPUT_SIZE, SEED


# ── Gabor filter bank ─────────────────────────────────────────────────────────
def _build_gabor_bank():
    """
    Returns list of Gabor kernels covering 8 orientations (0°–157.5°).
    Fingerprint ridges run in specific directions — Gabor captures this.
    """
    kernels = []
    ksize   = 31          # kernel size (must be odd)
    sigma   = 4.0         # gaussian envelope width
    lam     = 10.0        # wavelength of the ridge (~500 DPI ridge spacing)
    gamma   = 0.5         # spatial aspect ratio
    psi     = 0           # phase offset
    for theta_deg in range(0, 180, 23):   # 8 orientations
        theta = theta_deg * np.pi / 180.0
        k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, psi,
                               ktype=cv2.CV_32F)
        k /= k.sum() + 1e-6   # normalise
        kernels.append(k)
    return kernels


_GABOR_BANK = _build_gabor_bank()   # build once at import time


def apply_gabor(img_uint8):
    """
    Apply Gabor bank to a uint8 grayscale image.
    Returns: max-response image (uint8), same size as input.
    Captures the strongest ridge orientation at each pixel.
    """
    responses = [cv2.filter2D(img_uint8, cv2.CV_32F, k) for k in _GABOR_BANK]
    stacked   = np.stack(responses, axis=0)           # (8, H, W)
    max_resp  = np.max(stacked, axis=0)               # (H, W) — max over orientations
    # Normalise to 0–255
    mn, mx = max_resp.min(), max_resp.max()
    if mx - mn < 1e-6:
        return img_uint8.copy()
    norm = ((max_resp - mn) / (mx - mn) * 255).astype(np.uint8)
    return norm


# ── Main preprocessing function ───────────────────────────────────────────────
def preprocess_single_image(img_input):
    """
    Full preprocessing pipeline.
    Accepts: file path (str), numpy array (BGR or gray), or PIL Image.
    Returns: float32 numpy array, shape (224, 224), values in [0, 1].
             Returns None if image cannot be loaded.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if isinstance(img_input, str):
        img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        import PIL.Image
        img = np.array(img_input.convert('L'))

    if img is None or img.size == 0:
        return None

    # ── Step 1: Orientation fix (portrait) ───────────────────────────────────
    h, w = img.shape
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # ── Step 2: CLAHE (local contrast enhancement) ───────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # ── Step 3: Gabor filter — ridge orientation extraction ──────────────────
    img = apply_gabor(img)

    # ── Step 4: Gaussian blur (reduce sensor noise) ───────────────────────────
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # ── Step 5: Resize to 224×224 ────────────────────────────────────────────
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # ── Step 6: Normalise to [0, 1] ──────────────────────────────────────────
    return img.astype(np.float32) / 255.0


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_image(img_array):
    """
    Returns a list of augmented versions of a preprocessed float32 (224,224) image.
    Includes: original + small rotations + brightness shifts + small zoom.
    NO horizontal flip — mirror fingerprints do not exist in nature.
    """
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)

    # Ensure 2D
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    h, w   = img_array.shape
    center = (w // 2, h // 2)
    results = [img_array]   # always include original

    # Small rotations (fingerprints are placed at slight angles)
    for angle in [-10, -7, -4, 4, 7, 10]:
        M   = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(img_array, M, (w, h),
                             borderMode=cv2.BORDER_REFLECT)
        results.append(rot)

    # Brightness (pressure variation changes ridge contrast)
    for factor in [0.88, 0.93, 1.07, 1.12]:
        results.append(np.clip(img_array * factor, 0.0, 1.0).astype(np.float32))

    # Zoom in/out ±5%
    for scale in [0.95, 1.05]:
        M    = cv2.getRotationMatrix2D(center, 0, scale)
        zoom = cv2.warpAffine(img_array, M, (w, h),
                              borderMode=cv2.BORDER_REFLECT)
        results.append(zoom)

    return results   # 1 original + 6 rotations + 4 brightness + 2 zoom = 13 total


if __name__ == "__main__":
    print("Preprocess module OK")
    print(f"  Gabor kernels : {len(_GABOR_BANK)}")
    print(f"  Input size    : {INPUT_SIZE}")
    print(f"  Aug versions  : 13 per image (no horizontal flip)")
