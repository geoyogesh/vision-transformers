"""Vision Transformer model.

Mirrors the classes in notebooks/01_vit_from_scratch.ipynb (same attribute names, so
state_dict keys match; tests/test_notebook_parity.py enforces this), but takes its
hyperparameters from ViTConfig instead of notebook globals.
"""

import torch
import torch.nn as nn

from vit.config import ViTConfig


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            cfg.num_channels, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, x):
        x = self.patch_embed(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        return x.transpose(1, 2)  # (B, N, E)


class TransformerArchitecture(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(cfg.embed_dim)
        self.self_attention = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(cfg.embed_dim)
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.mlp_dim),
            nn.GELU(),
            nn.Linear(cfg.mlp_dim, cfg.embed_dim),
        )

    def forward(self, x):
        h = self.layer_norm_1(x)
        x = x + self.self_attention(h, h, h, need_weights=False)[0]
        x = x + self.multi_layer_perceptron(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig | None = None):
        super().__init__()
        cfg = cfg or ViTConfig()
        self.cfg = cfg
        self.patch_embedding = PatchEmbedding(cfg)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.num_patches + 1, cfg.embed_dim))
        self.transformer_layers = nn.Sequential(
            *[TransformerArchitecture(cfg) for _ in range(cfg.transformer_units)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer_layers(x)
        return self.mlp_head(x[:, 0])


def save_checkpoint(model: VisionTransformer, path: str, **metadata) -> None:
    """Save config + weights + arbitrary metadata (metrics, provenance)."""
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(), **metadata}, path)


def load_checkpoint(
    path: str, device: str | torch.device = "cpu"
) -> tuple[VisionTransformer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = VisionTransformer(ViTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
