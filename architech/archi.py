import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 1. Shared Stem
# -------------------------------------------------
class SharedStem(nn.Module):
    """
    Stem chung: trích feature global để router dùng.
    Có thể cho CNN expert dùng lại feature map này.
    """
    def __init__(self, in_chans=3, stem_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, stem_dim, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(stem_dim)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return:
          feat_global: (B, stem_dim)  - dùng cho router
          feat_map:    (B, stem_dim, H/2, W/2) - CNN expert dùng nếu muốn
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        feat_map = x                          # (B, stem_dim, H/2, W/2)
        feat_global = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, stem_dim)
        return feat_global, feat_map


# -------------------------------------------------
# 2. CNN Expert (ResNet mini skeleton)
# -------------------------------------------------
class CNNExpert(nn.Module):
    """
    CNN nhỏ: nhận ảnh (hoặc feature map từ stem) -> embedding (B, embed_dim)
    Ở đây dùng luôn ảnh gốc cho đơn giản, bạn có thể chỉnh lại để dùng feat_map.
    """
    def __init__(self, in_chans=3, embed_dim=192):
        super().__init__()
        # Một CNN rất đơn giản, bạn có thể thay bằng ResNet-18 / convstack
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.act   = nn.ReLU(inplace=True)

        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: (B, embed_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 128)
        x = self.fc(x)                              # (B, embed_dim)
        return x


# -------------------------------------------------
# 3. BERT-tiny-style Image Expert
#    (patch embedding + encoder giống BERT nhỏ)
# -------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Cùng dạng PatchEmbed như ViT
    (B, 3, H, W) -> (B, N, D)
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)         # (B, D, H/P, W/P)
        x = x.flatten(2)         # (B, D, N)
        x = x.transpose(1, 2)    # (B, N, D)
        return x


class BertTinyBlock(nn.Module):
    """
    Block encoder kiểu BERT:
    LN -> MHSA -> add -> LN -> FFN -> add
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1   = nn.Linear(dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, dim)
        self.act   = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout1(x)
        x = x + h

        # FFN
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
    """
    BERT-tiny cho ảnh: patch embedding + vài BertTinyBlock
    Output: CLS embedding (B, embed_dim)
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BertTinyBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # init đơn giản
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: (B, embed_dim)
        """
        B = x.size(0)
        x = self.patch_embed(x)              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)       # (B, N+1, D)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_embed = x[:, 0]                  # (B, D)
        return cls_embed


# -------------------------------------------------
# 4. ViT Expert (dùng model ViT bạn đã có)
# -------------------------------------------------

# --- RESTORED CLASSES (MLP, MultiHeadSelfAttention, EncoderBlock, MoEFFN, MoEEncoderBlock, VisionMoEViT) ---

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.drop_path_attn = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dropout=dropout)
        self.drop_path_mlp = nn.Dropout(dropout)

    def forward(self, x):
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
    def __init__(self, dim, hidden_dim, num_experts=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(dim, hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
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
            y_e = y_e * score_e
            y_flat[mask] = y_e

        y = y_flat.view(B, L, D)
        return y

class MoEEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, num_experts=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.drop_path_attn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.moe_ffn = MoEFFN(dim, hidden_dim, num_experts=num_experts, dropout=dropout)
        self.drop_path_mlp = nn.Dropout(dropout)

    def forward(self, x):
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
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        num_experts=4,
        moe_layers=(2, 3),
        dropout=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        blocks = []
        for i in range(depth):
            if i in moe_layers:
                block = MoEEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_experts=num_experts, dropout=dropout)
            else:
                block = EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            blocks.append(block)
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

    def forward_features(self, x):
        """
        Trả về feature embedding (CLS token) trước khi vào head.
        """
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

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.head(x)
        return logits

# -------------------------------------------------
# 5. ViT Expert Wrapper
# -------------------------------------------------
class ViTExpert(nn.Module):
    """
    Wrapper quanh ViT hiện có để trả về embedding (trước head).
    """
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        # Bây giờ VisionMoEViT đã có forward_features
        if hasattr(self.vit, "forward_features"):
            cls_embed = self.vit.forward_features(x)
        else:
            # Fallback nếu dùng model khác không có forward_features
            # (hoặc báo lỗi)
            raise NotImplementedError("Model ViT cần có hàm forward_features")
        return cls_embed

class HybridBackboneMoE(nn.Module):
    """
    MoE cấp backbone:
      - Stem chung: SharedStem
      - Router: sử dụng feat_global từ stem -> gate_probs (B, 3)
      - 3 expert:
          0: CNNExpert
          1: BertTinyImageExpert
          2: ViTExpert
      - Output: embedding (B, embed_dim), gate_probs (B, 3)
    """
    def __init__(
        self,
        img_size=32,
        in_chans=3,
        embed_dim=192,
        stem_dim=64,
        # cấu hình cho expert:
        bert_depth=4,
        bert_heads=3,
        vit_model=None,  # truyền vào instance ViT đã build sẵn
    ):
        super().__init__()

        # Stem + router
        self.stem = SharedStem(in_chans=in_chans, stem_dim=stem_dim)
        self.router = nn.Linear(stem_dim, 3)   # 3 expert
        self.softmax = nn.Softmax(dim=-1)

        # CNN expert
        self.cnn_expert = CNNExpert(in_chans=in_chans, embed_dim=embed_dim)

        # BERT-tiny-image expert
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

        # ViT expert (dùng model bạn đã có)
        assert vit_model is not None, "Bạn cần truyền vit_model (VisionMoEViT hoặc ViT) vào."
        self.vit_expert = ViTExpert(vit_model)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        return:
          h: (B, embed_dim)  - embedding đã mix từ 3 expert
          gate_probs: (B, 3) - phân phối expert cho từng sample
        """
        B = x.size(0)

        # 1) Stem + router
        stem_global, _ = self.stem(x)         # (B, stem_dim), (_ : feat_map)
        gate_logits = self.router(stem_global)   # (B, 3)
        gate_probs  = self.softmax(gate_logits)  # (B, 3)

        # 2) Chạy 3 expert
        h_cnn  = self.cnn_expert(x)       # (B, D)
        h_bert = self.bert_expert(x)      # (B, D)
        h_vit  = self.vit_expert(x)       # (B, D)

        # 3) Soft mixture theo gate_probs
        p_cnn  = gate_probs[:, 0].unsqueeze(-1)   # (B, 1)
        p_bert = gate_probs[:, 1].unsqueeze(-1)
        p_vit  = gate_probs[:, 2].unsqueeze(-1)

        h = p_cnn * h_cnn + p_bert * h_bert + p_vit * h_vit  # (B, D)
        return h, gate_probs

class HybridClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10, embed_dim=192):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # backbone returns (embedding, gate_probs)
        embedding, _ = self.backbone(x)
        logits = self.head(embedding)
        return logits
