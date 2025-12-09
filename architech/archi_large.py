import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# Text encoder (Transformer-based)
# ----------------------------------------
class MoEFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, dropout: float = 0.1, expert_devices: List[str] | None = None):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.expert_devices = expert_devices or [None] * num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])
        for expert, dev in zip(self.experts, self.expert_devices):
            if dev is not None:
                expert.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        base_device = x_flat.device
        gate_probs = torch.softmax(self.gate(x_flat), dim=-1)
        expert_idx = torch.argmax(gate_probs, dim=-1)
        expert_score = gate_probs.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)
        y_flat = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (expert_idx == e)
            if not mask.any():
                continue
            x_e = x_flat[mask]
            target_device = self.expert_devices[e]
            if target_device is not None and target_device != base_device:
                x_e = x_e.to(target_device, non_blocking=True)
                y_e = self.experts[e](x_e).to(base_device, non_blocking=True)
            else:
                y_e = self.experts[e](x_e)
            y_flat[mask] = y_e * expert_score[mask].unsqueeze(-1)
        return y_flat.view(B, L, D)


class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, num_experts: int = 4, expert_devices: List[str] | None = None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.moe_ffn = MoEFFN(d_model, dim_feedforward, num_experts=num_experts, dropout=dropout, expert_devices=expert_devices)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        h = src
        src = self.norm1(src)
        src, _ = self.self_attn(src, src, src)
        src = self.dropout1(src) + h

        h = src
        src = self.norm2(src)
        src = self.moe_ffn(src)
        src = self.dropout2(src) + h
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1), :]


class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, nhead: int, num_layers: int, dim_feedforward: int, num_experts: int = 4, dropout: float = 0.1, expert_devices: List[str] | None = None, max_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len + 1)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layers = []
        for _ in range(num_layers):
            layers.append(MoETransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_experts=num_experts,
                expert_devices=expert_devices,
            ))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        B, L = token_ids.shape
        x = self.embed(token_ids)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, D)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # CLS pooling
        return cls_out


# ----------------------------------------
# Generator building blocks
# ----------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        h = self.skip(h)
        return F.relu(x + h)


class FiLM(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, in_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_ch, out_ch)
        self.film = FiLM(out_ch, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.res(x)
        x = self.film(x, cond)
        return x


class LargeGenerator(nn.Module):
    """
    Bigger text-conditioned generator (UNet-ish upsampling stack with FiLM).
    target_size must be a power-of-two multiple of start_size.
    """
    def __init__(
        self,
        text_dim: int,
        latent_dim: int = 128,
        base_channels: int = 256,
        img_channels: int = 3,
        target_size: int = 256,
        start_size: int = 4,
    ):
        super().__init__()
        if target_size % start_size != 0:
            raise ValueError("target_size must be divisible by start_size")
        up_factor = target_size // start_size
        if up_factor & (up_factor - 1) != 0:
            raise ValueError("target_size/start_size must be a power of two")

        self.latent_dim = latent_dim
        self.start_size = start_size

        self.fc = nn.Linear(text_dim + latent_dim, base_channels * start_size * start_size)

        # Build upsampling stack
        up_blocks: List[nn.Module] = []
        channels = base_channels
        for _ in range(int(math.log2(up_factor))):
            out_ch = max(img_channels, channels // 2)
            up_blocks.append(UpBlock(channels, out_ch, cond_dim=text_dim))
            channels = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)
        self.to_rgb = nn.Conv2d(channels, img_channels, kernel_size=3, padding=1)

    def forward(self, text_feat: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z = torch.cat([text_feat, noise], dim=1)
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.start_size, self.start_size)
        for block in self.up_blocks:
            x = block(x, text_feat)
        x = torch.tanh(self.to_rgb(x))
        return x


# ----------------------------------------
# Full model
# ----------------------------------------
@dataclass
class LargeT2IConfig:
    vocab_size: int = 4096
    max_len: int = 32
    text_embed_dim: int = 512
    text_heads: int = 8
    text_layers: int = 6
    text_ff: int = 1024
    text_num_experts: int = 4
    text_expert_devices: List[str] | None = None  # e.g., ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    latent_dim: int = 128
    base_channels: int = 256
    target_size: int = 256


class LargeTextToImageModel(nn.Module):
    def __init__(self, cfg: LargeT2IConfig):
        super().__init__()
        self.text_encoder = TransformerTextEncoder(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.text_embed_dim,
            nhead=cfg.text_heads,
            num_layers=cfg.text_layers,
            dim_feedforward=cfg.text_ff,
            num_experts=cfg.text_num_experts,
            expert_devices=cfg.text_expert_devices,
        )
        self.generator = LargeGenerator(
            text_dim=cfg.text_embed_dim,
            latent_dim=cfg.latent_dim,
            base_channels=cfg.base_channels,
            target_size=cfg.target_size,
        )
        self.latent_dim = cfg.latent_dim
        self.max_len = cfg.max_len

    def forward(self, token_ids: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        text_feat = self.text_encoder(token_ids)
        return self.generator(text_feat, noise)
