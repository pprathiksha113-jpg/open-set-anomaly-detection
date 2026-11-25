import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.models.feature_extractor import FeatureExtractor


def greedy_coreset(feats, n_select, device, proj_dim=128):
    """Pick n_select patches that cover the normal feature space (k-center greedy)."""
    N, C = feats.shape
    proj = torch.randn(C, proj_dim, device=device) / math.sqrt(proj_dim)
    z = feats.to(device) @ proj
    idx = int(torch.randint(0, N, (1,)))
    selected = [idx]
    min_d = ((z - z[idx]) ** 2).sum(1)
    for _ in tqdm(range(n_select - 1), desc="coreset"):
        idx = int(torch.argmax(min_d))
        selected.append(idx)
        min_d = torch.minimum(min_d, ((z - z[idx]) ** 2).sum(1))
    return torch.tensor(selected)


class PatchCore:
    def __init__(self, coreset_ratio=0.1, k=1, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor().to(self.device)
        self.coreset_ratio = coreset_ratio
        self.k = k
        self.bank = None  # memory bank of NORMAL patches

    @torch.no_grad()
    def fit(self, loader):
        all_feats = []
        for batch in tqdm(loader, desc="extract normal features"):
            f = self.extractor(batch["image"].to(self.device))
            all_feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu())
        feats = torch.cat(all_feats)
        n = max(1, int(len(feats) * self.coreset_ratio))
        idx = greedy_coreset(feats, n, self.device)
        self.bank = feats[idx].to(self.device)
        print(f"memory bank: {tuple(self.bank.shape)}")

    @torch.no_grad()
    def predict(self, images, out_size=256):
        """Returns (anomaly_maps B x H x W, image_scores B)."""
        f = self.extractor(images.to(self.device))
        B, C, H, W = f.shape
        patches = f.permute(0, 2, 3, 1).reshape(-1, C)
        d = torch.cdist(patches, self.bank)
        d = d.topk(self.k, dim=1, largest=False).values.mean(1)
        amap = d.reshape(B, 1, H, W)
        amap = F.interpolate(amap, size=out_size, mode="bilinear", align_corners=False)
        amap = TF.gaussian_blur(amap, kernel_size=33, sigma=4.0).squeeze(1)
        scores = amap.flatten(1).max(1).values
        return amap.cpu(), scores.cpu()

    def save(self, path):
        torch.save(self.bank.cpu(), path)

    def load(self, path):
        self.bank = torch.load(path).to(self.device)