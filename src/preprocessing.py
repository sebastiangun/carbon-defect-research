"""Image representations used in CarbonDefect experiments.

Core HEDCM pipeline extracted from the current research Colab notebook.
Images are expected in OpenCV BGR format.
"""

import cv2
import numpy as np

HEDCM_PARAMS = {
    "HSV_GAMMA": 2.4,
    "HSV_CLIP_PERCENT": 0.5,
    "HSV_DARKEN_FACTOR": 0.75,
    "MEDIAN_BLUR_SIZE": 15,
    "CLAHE_CLIP": 2.5,
    "CLAHE_TILE": (8, 8),
    "DEFECT_GAMMA": 0.70,
    "ROBUST_LOW": 1.0,
    "ROBUST_HIGH": 99.0,
}


def homogenize_and_darken(img_bgr, factor):
    if factor is None:
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]

    sigma = max(15.0, min(img_bgr.shape[:2]) / 18.0)
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma, sigmaY=sigma)

    target = float(np.median(background))
    normalized = lightness / (background + 1.0) * target
    mixed = 0.70 * normalized + 0.30 * lightness

    lab[:, :, 0] = np.clip(mixed * float(factor), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def hedcm_stages(img_bgr):
    """Return intermediate HEDCM stages and final pseudocolor representation."""
    p = HEDCM_PARAMS

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    value = np.power(np.clip(value, 0, 1), p["HSV_GAMMA"])

    low = np.percentile(value, p["HSV_CLIP_PERCENT"])
    high = np.percentile(value, 100 - p["HSV_CLIP_PERCENT"])
    if high > low:
        value = np.clip((value - low) / (high - low), 0, 1)

    value = np.clip(value * p["HSV_DARKEN_FACTOR"], 0, 1)
    hsv[:, :, 2] = (value * 255).astype(np.uint8)

    darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(darkened, cv2.COLOR_BGR2GRAY)

    kernel = int(p["MEDIAN_BLUR_SIZE"])
    kernel += kernel % 2 == 0
    background = cv2.medianBlur(gray, kernel)

    deviation = cv2.absdiff(gray, background).astype(np.float32)

    dlo = np.percentile(deviation, p["ROBUST_LOW"])
    dhi = np.percentile(deviation, p["ROBUST_HIGH"])
    if dhi > dlo:
        normalized = np.clip((deviation - dlo) / (dhi - dlo), 0, 1)
    else:
        normalized = np.zeros_like(deviation)

    normalized_u8 = (normalized * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=p["CLAHE_CLIP"],
        tileGridSize=p["CLAHE_TILE"],
    )
    defect = clahe.apply(normalized_u8).astype(np.float32) / 255.0
    defect = np.power(np.clip(defect, 0, 1), p["DEFECT_GAMMA"])
    defect_u8 = (defect * 255).astype(np.uint8)

    hedcm = cv2.applyColorMap(255 - defect_u8, cv2.COLORMAP_TURBO)

    return {
        "darkened": darkened,
        "background": background,
        "deviation": deviation,
        "defect_map": defect_u8,
        "hedcm": hedcm,
    }


def build_representation(img_bgr, representation):
    """Build one of the six representations used in the benchmark."""
    representation = str(representation)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if representation == "RGB":
        return img_bgr.copy()
    if representation == "Grayscale":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if representation == "Viridis":
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    if representation == "Inferno":
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    if representation == "Turbo":
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if representation == "HEDCM":
        return hedcm_stages(img_bgr)["hedcm"]

    raise ValueError(f"Unknown representation: {representation}")
