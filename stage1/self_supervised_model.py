from typing import Dict

import timm
import torch
from torch import nn


class SelfSupervisedDegradationEncoder(nn.Module):
    """Anonymous degradation slots learned without fixed degradation-score targets."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        latent_dim: int = 128,
        num_slots: int = 4,
        num_heads: int = 4,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")
        if num_slots < 1:
            raise ValueError("num_slots must be positive.")

        self.backbone_name = backbone
        self.latent_dim = latent_dim
        self.num_slots = num_slots
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"Backbone '{backbone}' does not expose num_features.")
        self.feature_dim = feature_dim

        self.image_projection = nn.Conv2d(feature_dim, latent_dim, kernel_size=1)
        self.slots = nn.Parameter(torch.randn(1, num_slots, latent_dim) * 0.02)
        slot_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(slot_layer, num_layers=decoder_layers)
        self.cross_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(latent_dim)
        self.global_norm = nn.LayerNorm(latent_dim)
        self.magnitude_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _spatial_feature(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.backbone.forward_features(image)
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4D spatial feature map, got {tuple(feature.shape)}")
        if feature.shape[1] != self.feature_dim and feature.shape[-1] == self.feature_dim:
            feature = feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial = self._spatial_feature(image)
        projected = self.image_projection(spatial)
        batch, _, height, width = projected.shape
        image_tokens = projected.flatten(2).transpose(1, 2)

        slot_queries = self.slot_encoder(self.slots.expand(batch, -1, -1))
        slot_tokens, attention = self.cross_attention(
            slot_queries,
            image_tokens,
            image_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        slot_tokens = self.slot_norm(slot_tokens + slot_queries)
        z_deg = self.global_norm(slot_tokens.mean(dim=1))
        magnitude = torch.sigmoid(self.magnitude_head(z_deg))
        attention_maps = attention.mean(dim=1).reshape(batch, self.num_slots, height, width)

        return {
            "z_deg": z_deg,
            "slot_tokens": slot_tokens,
            "slot_attention": attention_maps,
            "m_deg": magnitude,
            "feature": spatial.mean(dim=(2, 3)),
            "spatial_feature": spatial,
        }


def load_self_supervised_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> SelfSupervisedDegradationEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = SelfSupervisedDegradationEncoder(
        backbone=config["backbone"],
        pretrained=False,
        latent_dim=config["latent_dim"],
        num_slots=config["num_slots"],
        num_heads=config["num_heads"],
        decoder_layers=config["decoder_layers"],
        dropout=config["dropout"],
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
