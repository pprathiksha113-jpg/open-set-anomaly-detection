import cv2
import numpy as np
import torch
from scipy.stats import ks_2samp

_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
STATS = ["brightness", "contrast", "sharpness", "noise"]
CAUSE = {"brightness": "lighting change", "contrast": "contrast / lighting change",
         "sharpness": "blur / focus change", "noise": "sensor noise"}


def image_stats(images):
    """Simple per-image statistics from normalized tensors."""
    arr = (images * _S + _M).clamp(0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)
    rows = []
    for im in arr:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        rows.append([g.mean(), g.std(), cv2.Laplacian(g, cv2.CV_32F).var(),
                     (g - cv2.GaussianBlur(g, (5, 5), 1.0)).std()])
    return np.array(rows)


@torch.no_grad()
def pooled_features(pc, images, bs=16):
    out = []
    for i in range(0, len(images), bs):
        f = pc.extractor(images[i:i + bs].to(pc.device))
        out.append(f.mean(dim=(2, 3)).cpu())
    return torch.cat(out).numpy()


def mmd_perm_test(X, Y, n_perm=500, seed=0):
    """MMD with RBF kernel (median heuristic) and a permutation p-value."""
    Z = np.vstack([X, Y])
    sq = (Z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    K = np.exp(-d2 / np.median(d2[d2 > 0]))
    n = len(X)

    def mmd(ix, iy):
        return K[np.ix_(ix, ix)].mean() + K[np.ix_(iy, iy)].mean() - 2 * K[np.ix_(ix, iy)].mean()

    idx = np.arange(len(Z))
    obs = mmd(idx[:n], idx[n:])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(idx)
        cnt += mmd(p[:n], p[n:]) >= obs
    return float((1 + cnt) / (1 + n_perm))


class ShiftDetector:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, images, pc):
        self.ref_stats = image_stats(images)
        self.ref_feats = pooled_features(pc, images)
        return self

    def detect(self, images, pc):
        stats = image_stats(images)
        feats = pooled_features(pc, images)
        ks_p = {n: float(ks_2samp(self.ref_stats[:, i], stats[:, i]).pvalue)
                for i, n in enumerate(STATS)}
        mmd_p = mmd_perm_test(self.ref_feats, feats)
        a = self.alpha / (len(STATS) + 1)  # Bonferroni
        flagged = [n for n, p in ks_p.items() if p < a]
        if flagged:
            cause = CAUSE[min(flagged, key=ks_p.get)]
        elif mmd_p < a:
            cause = "appearance / texture change"
        else:
            cause = None
        shifted = cause is not None
        return {"shifted": shifted, "cause": cause, "ks_p": ks_p, "mmd_p": mmd_p,
                "summary": f"shifted={shifted} | cause={cause} | mmd_p={mmd_p:.3f}"}