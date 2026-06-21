from typing import Dict

import timm
import torch
from torch import nn


class DegradationAssessor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        latent_dim: int = 128,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"Backbone '{backbone}' does not expose num_features.")

        self.token_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature = self.backbone(image)
        token = self.token_head(feature)
        scores = torch.sigmoid(self.score_head(feature))
        return {"scores": scores, "z_deg": token, "feature": feature}


def load_assessor(checkpoint_path: str, device: torch.device) -> DegradationAssessor:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = DegradationAssessor(
        backbone=config["backbone"],
        pretrained=False,
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

