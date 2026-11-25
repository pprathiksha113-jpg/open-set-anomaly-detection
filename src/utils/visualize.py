from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def save_panel(image, amap, pred_mask, gt_mask, title, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].imshow(image); ax[0].set_title("input")
    ax[1].imshow(image); ax[1].imshow(amap, cmap="jet", alpha=0.5); ax[1].set_title("anomaly heatmap")
    ax[2].imshow(pred_mask, cmap="gray"); ax[2].set_title("predicted mask")
    ax[3].imshow(gt_mask, cmap="gray"); ax[3].set_title("ground truth")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)