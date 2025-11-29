import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 1. Shared Stem
# -------------------------------------------------
class SharedStem(nn.Module):
    """
    Lightweight stem producing global features for routing and a feature map if needed.
    """
    def __init__(self, in_chans: int = 3, stem_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, stem_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(stem_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        feat_map = x
        feat_global = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return feat_global, feat_map


# -------------------------------------------------
# 2. CNN Expert
# -------------------------------------------------
class CNNExpert(nn.Module):
    """
    Small CNN that maps an image to an embedding.
    """
    def __init__(self, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)
        return x


# -------------------------------------------------
# 3. BERT-tiny-style Image Expert
# -------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class BertTinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout1(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + h
        return x


class BertTinyImageExpert(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([BertTinyBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


# -------------------------------------------------
# 4. ViT Expert (MoE-ready)
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.drop_path_attn = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dropout=dropout)
        self.drop_path_mlp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path_attn(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path_mlp(x)
        x = x + h
        return x


class MoEFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(dim, hidden_dim, dropout=dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_idx = torch.argmax(gate_probs, dim=-1)
        expert_score = gate_probs.gather(dim=-1, index=expert_idx.unsqueeze(-1)).squeeze(-1)
        y_flat = torch.zeros_like(x_flat)

        for e in range(self.num_experts):
            mask = (expert_idx == e)
            if not mask.any():
                continue
            x_e = x_flat[mask]
            y_e = self.experts[e](x_e)
            score_e = expert_score[mask].unsqueeze(-1)
            y_flat[mask] = y_e * score_e

        y = y_flat.view(B, L, D)
        return y


class MoEEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, num_experts: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.drop_path_attn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.moe_ffn = MoEFFN(dim, hidden_dim, num_experts=num_experts, dropout=dropout)
        self.drop_path_mlp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path_attn(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.moe_ffn(x)
        x = self.drop_path_mlp(x)
        x = x + h
        return x


class VisionMoEViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_experts: int = 4,
        moe_layers: Tuple[int, ...] = (2, 3),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        blocks: List[nn.Module] = []
        for i in range(depth):
            if i in moe_layers:
                blocks.append(MoEEncoderBlock(embed_dim, num_heads, mlp_ratio, num_experts=num_experts, dropout=dropout))
            else:
                blocks.append(EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        logits = self.head(x)
        return logits


# -------------------------------------------------
# 5. ViT Expert Wrapper
# -------------------------------------------------
class ViTExpert(nn.Module):
    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.vit = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vit, "forward_features"):
            return self.vit.forward_features(x)
        raise NotImplementedError("Model ViT cần có hàm forward_features")


class HybridBackboneMoE(nn.Module):
    """
    Hybrid MoE backbone with optional expert placement across devices.
    """
    def __init__(
        self,
        img_size: int = 32,
        in_chans: int = 3,
        embed_dim: int = 192,
        stem_dim: int = 64,
        bert_depth: int = 4,
        bert_heads: int = 3,
        vit_model: Optional[nn.Module] = None,
        expert_devices: Optional[Dict[str, str]] = None,  # {"cnn": "cuda:0", "bert": "cuda:0", "vit": "cuda:1"}
    ):
        super().__init__()
        self.stem = SharedStem(in_chans=in_chans, stem_dim=stem_dim)
        self.router = nn.Linear(stem_dim, 3)
        self.softmax = nn.Softmax(dim=-1)

        self.cnn_expert = CNNExpert(in_chans=in_chans, embed_dim=embed_dim)
        self.bert_expert = BertTinyImageExpert(
            img_size=img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=bert_depth,
            num_heads=bert_heads,
            mlp_ratio=4.0,
            dropout=0.1,
        )
        assert vit_model is not None, "Bạn cần truyền vit_model (VisionMoEViT hoặc ViT) vào."
        self.vit_expert = ViTExpert(vit_model)

        self.expert_devices = expert_devices or {}
        for name, device in self.expert_devices.items():
            if device is not None and hasattr(self, f"{name}_expert"):
                getattr(self, f"{name}_expert").to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_device = x.device

        # Stem + router
        stem_global, _ = self.stem(x)
        gate_logits = self.router(stem_global)
        gate_probs = self.softmax(gate_logits)

        def run_expert(module: nn.Module, inp: torch.Tensor, target_device: Optional[str]) -> torch.Tensor:
            if target_device and target_device != base_device:
                out = module(inp.to(target_device, non_blocking=True))
                return out.to(base_device, non_blocking=True)
            return module(inp)

        h_cnn = run_expert(self.cnn_expert, x, self.expert_devices.get("cnn"))
        h_bert = run_expert(self.bert_expert, x, self.expert_devices.get("bert"))
        h_vit = run_expert(self.vit_expert, x, self.expert_devices.get("vit"))

        p_cnn = gate_probs[:, 0].unsqueeze(-1)
        p_bert = gate_probs[:, 1].unsqueeze(-1)
        p_vit = gate_probs[:, 2].unsqueeze(-1)

        h = p_cnn * h_cnn + p_bert * h_bert + p_vit * h_vit
        return h, gate_probs


class HybridClassifier(nn.Module):
    def __init__(self, backbone: HybridBackboneMoE, num_classes: int = 10, embed_dim: int = 192):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.backbone(x)
        logits = self.head(embedding)
        return logits


# -------------------------------------------------
# 6. Simple config + factory for a tiny model
# -------------------------------------------------
@dataclass
class HybridConfig:
    img_size: int = 32
    in_chans: int = 3
    num_classes: int = 100
    embed_dim: int = 128
    stem_dim: int = 48
    bert_depth: int = 2
    bert_heads: int = 2
    vit_depth: int = 4
    vit_heads: int = 4
    vit_mlp_ratio: float = 3.0
    vit_num_experts: int = 2
    vit_moe_layers: tuple = (1,)
    dropout: float = 0.1
    expert_devices: Optional[Dict[str, str]] = None


def build_hybrid_classifier(cfg: HybridConfig) -> HybridClassifier:
    vit_model = VisionMoEViT(
        img_size=cfg.img_size,
        patch_size=4,
        in_chans=cfg.in_chans,
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_heads,
        mlp_ratio=cfg.vit_mlp_ratio,
        num_experts=cfg.vit_num_experts,
        moe_layers=cfg.vit_moe_layers,
        dropout=cfg.dropout,
    )

    backbone = HybridBackboneMoE(
        img_size=cfg.img_size,
        in_chans=cfg.in_chans,
        embed_dim=cfg.embed_dim,
        stem_dim=cfg.stem_dim,
        bert_depth=cfg.bert_depth,
        bert_heads=cfg.bert_heads,
        vit_model=vit_model,
        expert_devices=cfg.expert_devices,
    )

    model = HybridClassifier(backbone, num_classes=cfg.num_classes, embed_dim=cfg.embed_dim)
    return model


# -------------------------------------------------
# 7. Lightweight text-to-image components (toy demo)
# -------------------------------------------------
def _stable_hash_token(token: str, vocab_size: int) -> int:
    h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
    return (h % (vocab_size - 1)) + 1


def tokenize_batch(texts, vocab_size: int = 4096, max_len: int = 16):
    token_seqs = []
    lengths = []
    for t in texts:
        tokens = t.lower().strip().split()
        if not tokens:
            tokens = ["unk"]
        ids = [_stable_hash_token(tok, vocab_size) for tok in tokens[:max_len]]
        length = len(ids)
        if length < max_len:
            ids.extend([0] * (max_len - length))
        token_seqs.append(ids)
        lengths.append(length)
    token_ids = torch.tensor(token_seqs, dtype=torch.long) 
    lengths = torch.tensor(lengths, dtype=torch.long)
    return token_ids, lengths


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        return h_n[-1]


class TextToImageGenerator(nn.Module):
    """
    Text embedding + noise -> image. Output size configurable; target_size/start_size must be power-of-two.
    """
    def __init__(
        self,
        text_dim: int = 256,
        latent_dim: int = 64,
        base_channels: int = 128,
        img_channels: int = 3,
        target_size: int = 32,
        start_size: int = 4,
    ):
        super().__init__()
        if target_size % start_size != 0:
            raise ValueError(f"target_size must be divisible by start_size ({start_size})")
        up_factor = target_size // start_size
        if up_factor & (up_factor - 1) != 0:
            raise ValueError("target_size/start_size must be a power of two")

        self.latent_dim = latent_dim
        self.target_size = target_size
        self.start_size = start_size

        num_upsamples = int(math.log2(up_factor))
        channels = base_channels
        self.fc = nn.Linear(text_dim + latent_dim, channels * start_size * start_size)

        layers: List[nn.Module] = []
        for _ in range(num_upsamples):
            out_ch = max(img_channels, channels // 2)
            layers.extend([
                nn.ConvTranspose2d(channels, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            channels = out_ch

        layers.append(nn.Conv2d(channels, img_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, text_feat: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z = torch.cat([text_feat, noise], dim=1)
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.start_size, self.start_size)
        img = self.net(x)
        return img


class TextToImageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        text_embed_dim: int = 128,
        text_hidden_dim: int = 256,
        latent_dim: int = 64,
        base_channels: int = 128,
        target_size: int = 32,
    ):
        super().__init__()
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=text_embed_dim, hidden_dim=text_hidden_dim)
        self.generator = TextToImageGenerator(
            text_dim=text_hidden_dim,
            latent_dim=latent_dim,
            base_channels=base_channels,
            target_size=target_size,
        )
        self.latent_dim = latent_dim

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        text_feat = self.text_encoder(token_ids, lengths)
        img = self.generator(text_feat, noise)
        return img
