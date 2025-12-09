import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------
# Modern Components: RMSNorm, RoPE, SwiGLU
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len: int):
        # x: [B, H, L, D]
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, L, D]
    # cos, sin: [1, 1, L, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        # SwiGLU: (Swish(xW1) * xW2) W3
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

# ---------------------------------------------------------------------------
# M2N2 / Slerp Utilities (Preserved)
# ---------------------------------------------------------------------------

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, dot_threshold: float = 0.9995) -> torch.Tensor:
    v0_norm = torch.norm(v0)
    v1_norm = torch.norm(v1)
    v0_n = v0 / (v0_norm + 1e-8)
    v1_n = v1 / (v1_norm + 1e-8)
    dot = torch.sum(v0_n * v1_n)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    if torch.abs(dot) > dot_threshold:
        return (1 - t) * v0 + t * v1
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res_n = s0 * v0_n + s1 * v1_n
    res_norm = (1 - t) * v0_norm + t * v1_norm
    return res_n * res_norm

def m2n2_merge_weights(w1: torch.Tensor, w2: torch.Tensor, ratio: float) -> torch.Tensor:
    w1_flat = w1.view(-1)
    w2_flat = w2.view(-1)
    merged_flat = slerp(w1_flat, w2_flat, ratio)
    return merged_flat.view_as(w1)

# ---------------------------------------------------------------------------
# Modernized MoE Layer with M2N2
# ---------------------------------------------------------------------------

class M2N2MoELayer(nn.Module):
    """
    MoE Layer with 'Competitive & Aggregation' (C&A) style Parameter Merging for ALL experts.
    
    Instead of selecting Top-K, we aggregate the parameters of ALL experts based on the gating weights.
    Mathematically:
    y = ( Activation( x @ Sum(p_i * W1_i) ) ) @ Sum(p_i * W3_i)
    
    Efficient Implementation:
    Since x @ Sum(p_i * W_i) == Sum(p_i * (x @ W_i)), we compute the linear projections 
    for all experts and then average them. This avoids materializing the huge merged weight tensor.
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Experts: SwiGLU structure (3 matrices per expert)
        # We use a single large tensor for efficient computation (Batch Matrix Multiply)
        # Shape: [NumExperts, In, Out]
        self.experts_w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.experts_w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.experts_w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.experts_w1)
        nn.init.xavier_uniform_(self.experts_w2)
        nn.init.xavier_uniform_(self.experts_w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, SeqLen, Dim]
        """
        B, L, D = x.shape
        
        # 1. Gating (Competition)
        # [B, L, D] @ [D, E] -> [B, L, E]
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1) # [B, L, E]
        
        # 2. Aggregation (Parameter Merging)
        # We want to compute: h = Activation( x @ W_merged )
        # W_merged = Sum(p_i * W_i)
        # x @ W_merged = Sum(p_i * (x @ W_i))
        
        # Reshape x for broadcasting: [B, L, 1, D]
        x_reshaped = x.unsqueeze(2) 
        
        # Compute all experts' linear projections efficiently
        # We can use einsum. 
        # x: [B, L, D]
        # W: [E, D, H]
        # Out: [B, L, E, H]
        
        # W1 projection
        # [B, L, D] x [E, D, H] -> [B, L, E, H]
        out_w1 = torch.einsum('bld,edh->bleh', x, self.experts_w1)
        
        # W2 projection (for SwiGLU gate)
        out_w2 = torch.einsum('bld,edh->bleh', x, self.experts_w2)
        
        # Aggregate Pre-Activations
        # Sum(p_i * out_i) -> [B, L, H]
        # gate_probs: [B, L, E]
        # out: [B, L, E, H]
        # result: [B, L, H]
        merged_w1 = torch.einsum('ble,bleh->blh', gate_probs, out_w1)
        merged_w2 = torch.einsum('ble,bleh->blh', gate_probs, out_w2)
        
        # 3. Activation (SwiGLU) applied on the MERGED representation
        # This is the key difference from standard MoE
        hidden = F.silu(merged_w1) * merged_w2
        
        # 4. Second Layer Aggregation
        # We need to compute: hidden @ Sum(p_i * W3_i)
        # = Sum(p_i * (hidden @ W3_i))
        
        # W3 projection for all experts
        # hidden: [B, L, H]
        # W3: [E, H, D]
        # Out: [B, L, E, D]
        out_w3 = torch.einsum('blh,ehd->bled', hidden, self.experts_w3)
        
        # Aggregate
        output = torch.einsum('ble,bled->bld', gate_probs, out_w3)
        
        output = self.dropout(output)
        return output

# ---------------------------------------------------------------------------
# Modern Attention Block
# ---------------------------------------------------------------------------

class ModernAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, L, D_head]
        
        if rotary_emb is not None:
            cos, sin = rotary_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
        # Flash Attention (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        return out

class ModernBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, num_experts: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = ModernAttention(dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        # 2/3 ratio often used with SwiGLU to keep param count similar to GELU
        hidden_dim = int(2 * hidden_dim / 3) 
        
        self.moe = M2N2MoELayer(dim, hidden_dim, num_experts=num_experts, dropout=dropout)

    def forward(self, x: torch.Tensor, rotary_emb: Optional[Tuple] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rotary_emb=rotary_emb)
        x = x + self.moe(self.norm2(x))
        return x

# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class ModernViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, num_experts=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # RoPE
        self.rope = RotaryEmbedding(embed_dim // num_heads)
        
        self.blocks = nn.ModuleList([
            ModernBlock(embed_dim, num_heads, mlp_ratio=4.0, num_experts=num_experts)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        self.apply(self._init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Prepare RoPE
        rotary_emb = self.rope(x, x.shape[1])
        
        for blk in self.blocks:
            x = blk(x, rotary_emb=rotary_emb)
            
        x = self.norm(x)
        return x

class ModernTextEncoder(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=768, depth=12, num_heads=12, max_len=512, num_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rope = RotaryEmbedding(embed_dim // num_heads, max_seq_len=max_len)
        
        self.blocks = nn.ModuleList([
            ModernBlock(embed_dim, num_heads, mlp_ratio=4.0, num_experts=num_experts)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        rotary_emb = self.rope(x, x.shape[1])
        
        for blk in self.blocks:
            x = blk(x, rotary_emb=rotary_emb)
            
        x = self.norm(x)
        return x

# ---------------------------------------------------------------------------
# Main Architecture
# ---------------------------------------------------------------------------

class ArchiMMoMoE(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16, in_chans=3,
        vocab_size=30522, max_len=512,
        embed_dim=512, depth=6, num_heads=8, num_experts=4
    ):
        super().__init__()
        
        self.image_encoder = ModernViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_experts=num_experts
        )
        
        self.text_encoder = ModernTextEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim, depth=depth, 
            num_heads=num_heads, max_len=max_len, num_experts=num_experts
        )
        
        # Projections for CLIP-style loss
        self.img_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        nn.init.trunc_normal_(self.img_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.text_proj.weight, std=0.02)

    def forward(self, image=None, text=None):
        img_feat = None
        text_feat = None
        
        if image is not None:
            img_out = self.image_encoder(image)
            # CLS token at index 0
            img_feat = img_out[:, 0]
            img_feat = self.img_proj(img_feat)
            
        if text is not None:
            text_out = self.text_encoder(text)
            # CLS token assumption (usually index 0 for BERT-style, but depends on tokenizer)
            # For causal/GPT style it might be last, but we stick to BERT-style here
            text_feat = text_out[:, 0]
            text_feat = self.text_proj(text_feat)
            
        return {
            "image_features": img_feat,
            "text_features": text_feat
        }

if __name__ == "__main__":
    model = ArchiMMoMoE()
    print("Modern ArchiMMoMoE initialized.")
    img = torch.randn(1, 3, 224, 224)
    txt = torch.randint(0, 1000, (1, 20))
    out = model(image=img, text=txt)
    print("Output shapes:", out["image_features"].shape, out["text_features"].shape)
