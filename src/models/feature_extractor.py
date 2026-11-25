import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FeatureExtractor(nn.Module):
    """Frozen WideResNet50 -> mid-level patch features (layer2 + layer3)."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # B x 1536 x 32 x 32 for 256 input