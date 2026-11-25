import torch


class FalsePositiveManager:
    """Three-way decision: pass / review / defect.
    Only confident cases are auto-rejected; borderline ones go to a human."""

    def __init__(self, calibrator, alpha_reject=0.05, alpha_review=0.15):
        self.cal = calibrator
        self.alpha_reject = alpha_reject
        self.alpha_review = alpha_review
        self.queue = []

    def decide(self, score):
        p = float(self.cal.p_value(score)[0])
        if p <= self.alpha_reject:
            return "defect"
        if p <= self.alpha_review:
            return "review"
        return "pass"

    def submit(self, name, score):
        d = self.decide(score)
        if d == "review":
            self.queue.append({"name": name, "score": float(score)})
        return d


@torch.no_grad()
def add_confirmed_normal(pc, images, top_m=200):
    """Feedback loop: after a human confirms an image is normal, add its most
    'surprising' patches to the memory bank so it stops being a false positive."""
    f = pc.extractor(images.to(pc.device))
    B, C, H, W = f.shape
    patches = f.permute(0, 2, 3, 1).reshape(-1, C)
    d = torch.cdist(patches, pc.bank).min(dim=1).values
    top = d.topk(min(top_m * B, len(d))).indices
    pc.bank = torch.cat([pc.bank, patches[top]])