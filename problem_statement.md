# Open-Set Industrial Anomaly Detection with Localization, Severity & Distribution Shift

## Goal
Detect defects the model has never seen. Train on normal samples only.

## Required components
1. Anomaly detection: image-level score (normal vs anomalous)
2. Localization: pixel-level heatmap and defect mask
3. Severity estimation: low / medium / high from area, intensity, and score
4. Confidence calibration: reliable probabilities, not raw scores
5. Distribution-shift detection: flag lighting, blur, noise, and texture changes
6. False-positive management: threshold tuning, review queue, normal-bank updates

## Evaluation
- Image AUROC, pixel AUROC, PRO score
- Expected calibration error (ECE)
- Shift detection accuracy
- False-positive rate at a fixed recall