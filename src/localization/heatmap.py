import cv2
import numpy as np


def fit_thresholds(val_scores, val_maps, q_img=0.99, q_pix=0.999):
    """Thresholds come from held-out NORMAL images only (open-set: no defect labels)."""
    return {
        "image": float(np.quantile(np.asarray(val_scores), q_img)),
        "pixel": float(np.quantile(np.asarray(val_maps).ravel(), q_pix)),
    }


def make_mask(amap, pix_thr, min_area=30):
    """Binary defect mask; tiny blobs are dropped as noise."""
    binary = (amap > pix_thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 1
    return mask