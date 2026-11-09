"""Backbones, classification head, freeze / unfreeze helpers."""
from __future__ import annotations

import timm
import torch
from torch import nn

# timm checkpoints for each architecture we compare
ARCHS = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3_large_100": "mobilenetv3_large_100.ra_in1k",
    "vit_small_patch14_dinov2": "vit_small_patch14_dinov2.lvd142m",
}


class SmallCNN(nn.Module):
    """Baseline B: four conv blocks trained from scratch."""

    def __init__(self, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            # GroupNorm, not BatchNorm: from scratch at this batch size, BN running stats lagged the weights and
            # eval-mode outputs saturated (all p > 0.99) while train-mode outputs were fine. GN has no train/eval gap.
            blocks += [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                       nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(chans[-1], num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class Classifier(nn.Module):
    """timm backbone (pooled features) -> dropout -> linear. Outputs one logit (positive = leukemic)."""

    def __init__(self, arch: str, pretrained: bool = True, dropout: float = 0.2, img_size: int = 224):
        super().__init__()
        kwargs = {"img_size": img_size} if "dinov2" in arch else {}
        self.backbone = timm.create_model(ARCHS.get(arch, arch), pretrained=pretrained, num_classes=0, **kwargs)
        # timm's num_features can differ from the pooled output (e.g. MobileNetV3 has a conv_head), so measure it.
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            feat_dim = self.backbone(torch.zeros(1, 3, img_size, img_size)).shape[-1]
            self.backbone.train(was_training)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        self.arch = arch

    def forward(self, x):
        return self.head(self.backbone(x))

    def features(self, x):
        return self.backbone(x)

    # ---- staged fine-tuning
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n: int) -> None:
        """Unfreeze the last `n` stages (CNN blocks or ViT blocks) plus the final norm layers."""
        self.freeze_backbone()
        bb = self.backbone
        if hasattr(bb, "blocks"):  # efficientnet / mobilenet / vit all expose .blocks in timm
            for blk in list(bb.blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
        for name in ("conv_head", "bn2", "norm", "fc_norm", "conv_stem_last"):
            mod = getattr(bb, name, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True


def build_model(cfg: dict) -> nn.Module:
    t = cfg["train"]
    if t["arch"] == "small_cnn":
        return SmallCNN(dropout=t["dropout"])
    return Classifier(t["arch"], pretrained=t["pretrained"], dropout=t["dropout"], img_size=cfg["data"]["img_size"])


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device, amp: bool = True) -> torch.Tensor:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().squeeze(1).cpu())
    return torch.cat(out)
