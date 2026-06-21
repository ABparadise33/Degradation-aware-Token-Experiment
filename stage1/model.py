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
        score_from_token: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.score_from_token = score_from_token
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise ValueError(f"Backbone '{backbone}' does not expose num_features.")

        self.token_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        if score_from_token:
            self.score_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 5),
            )
        else:
            # V1 checkpoint-compatible baseline.
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
        token, scores = self.score_pooled_features(feature)
        return {"scores": scores, "z_deg": token, "feature": feature}

    def score_pooled_features(self, feature: torch.Tensor):
        token = self.token_head(feature)
        score_input = token if self.score_from_token else feature
        scores = torch.sigmoid(self.score_head(score_input))
        return token, scores


class TaskAwareDegradationAssessor(nn.Module):
    """SFIQA-style task tokens attending to the final spatial backbone feature map."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        latent_dim: int = 128,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        num_tasks: int = 5,
        num_heads: int = 4,
        decoder_layers: int = 1,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")
        self.backbone_name = backbone
        self.num_tasks = num_tasks
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
        self.task_tokens = nn.Parameter(torch.randn(1, num_tasks, latent_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.task_encoder = nn.TransformerEncoder(encoder_layer, num_layers=decoder_layers)
        self.cross_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.task_norm = nn.LayerNorm(latent_dim)
        self.score_heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(num_tasks)])
        self.global_norm = nn.LayerNorm(latent_dim)

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

    def _decode_image_tokens(self, image_tokens: torch.Tensor):
        batch = image_tokens.shape[0]
        task_queries = self.task_encoder(self.task_tokens.expand(batch, -1, -1))
        task_tokens, attention = self.cross_attention(
            task_queries,
            image_tokens,
            image_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        task_tokens = self.task_norm(task_tokens + task_queries)
        scores = torch.cat(
            [head(task_tokens[:, index]) for index, head in enumerate(self.score_heads)],
            dim=1,
        ).sigmoid()
        global_token = self.global_norm(task_tokens.mean(dim=1))
        return task_tokens, global_token, scores, attention.mean(dim=1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial = self._spatial_feature(image)
        projected = self.image_projection(spatial)
        batch, channels, height, width = projected.shape
        image_tokens = projected.flatten(2).transpose(1, 2)
        task_tokens, global_token, scores, attention = self._decode_image_tokens(image_tokens)
        return {
            "scores": scores,
            "z_deg": global_token,
            "task_tokens": task_tokens,
            "attention_maps": attention.reshape(batch, self.num_tasks, height, width),
            "feature": spatial.mean(dim=(2, 3)),
            "spatial_feature": spatial,
        }

    def score_pooled_features(self, feature: torch.Tensor):
        projected = torch.nn.functional.linear(
            feature,
            self.image_projection.weight[:, :, 0, 0],
            self.image_projection.bias,
        ).unsqueeze(1)
        task_tokens, global_token, scores, _ = self._decode_image_tokens(projected)
        return global_token, scores, task_tokens


def load_assessor(checkpoint_path: str, device: torch.device) -> DegradationAssessor:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    architecture = config.get("architecture", "token_mlp")
    if architecture == "task_attention":
        model = TaskAwareDegradationAssessor(
            backbone=config["backbone"],
            pretrained=False,
            latent_dim=config["latent_dim"],
            dropout=config["dropout"],
            freeze_backbone=False,
            num_heads=config.get("num_heads", 4),
            decoder_layers=config.get("decoder_layers", 1),
        )
    else:
        model = DegradationAssessor(
            backbone=config["backbone"],
            pretrained=False,
            latent_dim=config["latent_dim"],
            dropout=config["dropout"],
            freeze_backbone=False,
            score_from_token=config.get("score_from_token", False),
        )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
