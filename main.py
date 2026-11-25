import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.patchcore import PatchCore
from src.utils.metrics import image_auroc, pixel_auroc


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    d, m = cfg["data"], cfg["model"]

    train = MVTecDataset(d["root"], d["category"], "train", d["image_size"])
    test = MVTecDataset(d["root"], d["category"], "test", d["image_size"])
    train_dl = DataLoader(train, batch_size=d["batch_size"], num_workers=0)
    test_dl = DataLoader(test, batch_size=d["batch_size"], num_workers=0)

    pc = PatchCore(coreset_ratio=m["coreset_ratio"], k=m["k_neighbors"])
    pc.fit(train_dl)
    pc.save(f"experiments/{d['category']}_bank.pt")

    labels, scores, masks, maps = [], [], [], []
    for b in test_dl:
        amap, s = pc.predict(b["image"])
        labels += b["label"].tolist()
        scores += s.tolist()
        masks.append(b["mask"].numpy())
        maps.append(amap.numpy())

    print("Image AUROC:", round(image_auroc(labels, scores), 4))
    print("Pixel AUROC:", round(pixel_auroc(np.concatenate(masks), np.concatenate(maps)), 4))


if __name__ == "__main__":
    main()