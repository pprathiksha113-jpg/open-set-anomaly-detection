from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """split='train' returns only normal images.
    split='test' returns normal + defective images with masks."""

    def __init__(self, root, category, split="train", image_size=256):
        self.root = Path(root) / category
        self.split = split
        self.size = image_size
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        self.samples = []  # (image_path, mask_path or None, label, defect_type)

        if split == "train":
            for p in sorted((self.root / "train" / "good").glob("*.png")):
                self.samples.append((p, None, 0, "good"))
        else:
            for d in sorted((self.root / "test").iterdir()):
                if not d.is_dir():
                    continue
                for p in sorted(d.glob("*.png")):
                    if d.name == "good":
                        self.samples.append((p, None, 0, "good"))
                    else:
                        m = self.root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                        self.samples.append((p, m, 1, d.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_p, mask_p, label, dtype = self.samples[i]
        img = self.tf(Image.open(img_p).convert("RGB"))
        if mask_p is not None and mask_p.exists():
            m = Image.open(mask_p).convert("L").resize((self.size, self.size))
            mask = torch.from_numpy((np.array(m) > 0).astype(np.float32))
        else:
            mask = torch.zeros(self.size, self.size)
        return {"image": img, "mask": mask, "label": label,
                "defect_type": dtype, "path": str(img_p)}