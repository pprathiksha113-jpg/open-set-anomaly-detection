import cv2
import numpy as np


def estimate_severity(amap, mask, pix_thr, area_cap=0.05):
    """Severity from defect area and how far the peak exceeds the threshold.
    Cutoffs are heuristic: tune them on your data."""
    if mask.sum() == 0:
        return {"level": "none", "score": 0.0, "area_ratio": 0.0, "regions": []}

    area_ratio = float(mask.mean())
    peak = float(amap[mask > 0].max())
    area_term = min(area_ratio / area_cap, 1.0)
    peak_term = float(np.clip((peak - pix_thr) / pix_thr, 0.0, 1.0))
    score = 0.5 * area_term + 0.5 * peak_term

    level = "low" if score < 0.25 else "medium" if score < 0.5 else "high"

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = [
        {"bbox": [int(s[0]), int(s[1]), int(s[2]), int(s[3])], "area": int(s[4])}
        for s in stats[1:n]
    ]
    return {"level": level, "score": round(score, 3),
            "area_ratio": round(area_ratio, 4), "regions": regions}