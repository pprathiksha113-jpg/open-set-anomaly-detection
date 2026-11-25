import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import MVTecDataset
from src.models.patchcore import PatchCore
from src.localization.heatmap import fit_thresholds, make_mask
from src.severity.severity import estimate_severity
from src.utils.metrics import image_auroc, pixel_auroc
from src.utils.visualize import denorm, save_panel


def predict_all(pc, loader):
    out = {"scores": [], "maps": [], "labels": [], "masks": [], "images": [], "types": []}
    for b in loader:
        amap, s = pc.predict(b["image"])
        out["scores"] += s.tolist()
        out["maps"].append(amap.numpy())
        out["labels"] += b["label"].tolist()
        out["masks"].append(b["mask"].numpy())
        out["images"].append(b["image"])
        out["types"] += list(b["defect_type"])
    out["maps"] = np.concatenate(out["maps"])
    out["masks"] = np.concatenate(out["masks"])
    out["images"] = torch.cat(out["images"])
    return out


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    d, m = cfg["data"], cfg["model"]
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # normal-only data: 90% builds the memory bank, 10% is held out for thresholds
    full = MVTecDataset(d["root"], d["category"], "train", d["image_size"])
    n_val = max(1, int(0.1 * len(full)))
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    test = MVTecDataset(d["root"], d["category"], "test", d["image_size"])

    mk = lambda ds: DataLoader(ds, batch_size=d["batch_size"], num_workers=0)

    pc = PatchCore(coreset_ratio=m["coreset_ratio"], k=m["k_neighbors"])
    pc.fit(mk(train_set))
    pc.save(f"experiments/{d['category']}_bank.pt")

    val = predict_all(pc, mk(val_set))
    res = predict_all(pc, mk(test))
    thr = fit_thresholds(val["scores"], val["maps"])
    print("thresholds:", thr)

    print("Image AUROC:", round(image_auroc(res["labels"], res["scores"]), 4))
    print("Pixel AUROC:", round(pixel_auroc(res["masks"], res["maps"]), 4))

    # masks + severity + result images for a few samples
    defect_idx = [i for i, l in enumerate(res["labels"]) if l == 1][:8]
    good_idx = [i for i, l in enumerate(res["labels"]) if l == 0][:2]
    for i in defect_idx + good_idx:
        mask = make_mask(res["maps"][i], thr["pixel"])
        sev = estimate_severity(res["maps"][i], mask, thr["pixel"])
        print(f"#{i} {res['types'][i]:<14} score={res['scores'][i]:.2f} "
              f"-> severity={sev['level']} ({sev['score']}) regions={len(sev['regions'])}")
        save_panel(denorm(res["images"][i]), res["maps"][i], mask, res["masks"][i],
                   f"{res['types'][i]} | severity: {sev['level']}",
                   f"reports/sample_{i}_{res['types'][i]}.png")


if __name__ == "__main__":
    main()