# PROMETHEUS-X
### Open-Set Industrial Anomaly Detection and Localization

**Problem Statement:** A06  
**Technology:** Artificial Intelligence | Computer Vision | Industrial Inspection

## 1. Project Overview

PROMETHEUS-X is an AI-based industrial visual inspection prototype designed to detect anomalous patterns in manufactured products.

The system uses PatchCore, a memory-bank-based anomaly detection approach, to learn normal visual patterns and identify deviations during inspection.

It provides anomaly scoring, defect localization through heatmaps, distribution-shift screening and a Streamlit-based inspection dashboard.

## 2. Technology Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Streamlit
- Plotly
- WideResNet50-2
- PatchCore
- MVTec AD Dataset

## 3. System Architecture

1. **Data Collection:** MVTec AD industrial bottle images.
2. **Preprocessing:** RGB conversion, resizing to 256 × 256 and ImageNet normalization.
3. **Feature Extraction:** Pretrained WideResNet50-2.
4. **Memory Bank Construction:** Representative normal image features are stored using a coreset.
5. **Anomaly Detection:** Test features are compared against normal memory-bank features using nearest-neighbour distances.
6. **Calibration:** A decision threshold is determined using held-out normal images.
7. **Localization:** Anomaly heatmaps highlight suspicious image regions.
8. **Visualization:** Results are displayed through the Streamlit dashboard.

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Model | PatchCore |
| Feature extractor | WideResNet50-2 |
| Training images | 167 normal bottle images |
| Calibration images | 42 normal bottle images |
| Input resolution | 256 × 256 |
| Coreset ratio | 0.005 |
| Nearest neighbours | 3 |
| Memory bank | 855 × 1,536 |
| Calibrated threshold | 1.85217 |
| GPU used | NVIDIA Tesla T4 |

## 5. Experimental Results

The team's recorded MVTec AD bottle test split contained 83 images.

| Evaluation metric | Result |
|---|---:|
| Image accuracy | 98.8% |
| Image AUROC | 1.0000 |
| Defect recall | 100% |
| Normal recall | 95% |
| Pixel AUROC | 0.9866 |

These results apply to the evaluated MVTec AD bottle test split and should not be interpreted as performance guarantees for other products or real-world production environments.

## 6. Dashboard Features

- Image-based anomaly inspection
- Live camera inspection prototype
- Anomaly heatmap visualization
- MVTec AD dataset explorer
- Inspection analytics and history
- System and model monitoring
- Distribution-shift screening

## 7. Repository Contents

- `src/` — Model, preprocessing, calibration and detection modules.
- `app/` — Streamlit dashboard.
- `train.py` — Memory-bank training entry point.
- `evaluate.py` — Model evaluation entry point.
- `tests/` — Automated tests.
- `notebooks/` — Training experiments.
- `artifacts/` — Local model artifacts (not necessarily included in Git).

## 8. Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the MVTec AD dataset separately and place it under `data/raw/mvtec/`.

Build the memory bank:

```bash
python train.py --data-root data/raw/mvtec --category bottle
```

Launch the dashboard after providing the required model artifacts:

```bash
streamlit run app/app.py
```

Evaluate the model:

```bash
python evaluate.py --data-root data/raw/mvtec --category bottle
```

## 9. Limitations and Future Scope

- Extend evaluation to additional industrial product categories.
- Improve robustness across lighting and camera variations.
- Develop human-in-the-loop memory-bank updates.
- Validate distribution-shift detection on real production data.
- Investigate deployment on edge computing devices.

Uploaded images are inspected; the current model does not automatically retrain itself.

## 10. Training Notebook

**Kaggle Training Notebook:**  
https://www.kaggle.com/code/muhammedrayyankhan/notebook21cde34cd5/edit

## 11. Project Repository

**GitHub:**  
https://github.com/pprathiksha113-jpg/open-set-anomaly-detection

---

**PROMETHEUS-X — Intelligent Visual Inspection for Industrial Quality Control.**
