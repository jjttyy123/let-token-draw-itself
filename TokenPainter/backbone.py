"""TokenPainter backbone: autoregressive decoder with RoPE + KV-cache.

Contains:
    KVCacheBlock  — Transformer block with GQA KV-cache, RoPE on Q/K
    RoPEDecoder   — Autoregressive decoder, z injected at every step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  Transformer Block with KV-cache + RoPE on Q/K
# ═══════════════════════════════════════════════════════════

class KVCacheBlock(nn.Module):
    """Transformer block with KV cache + RoPE on Q/K. Supports GQA (kv_heads < heads).

    RoPE is applied to Q and K individually after projection (standard formulation),
    giving the relative-position property: Q_i^T K_j = f(content, i-j).
    """

    def __init__(self, dim, heads=8, kv_heads=8, mlp_ratio=4.0, dropout=0.1,
                 max_pos=256):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.kv_dim = self.head_dim * kv_heads
        self.q_per_kv = heads // kv_heads

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.kv_dim)
        self.v_proj = nn.Linear(dim, self.kv_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

        # ── RoPE position encoding (per-block, based on head_dim) ──
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_pos).float()
        angles = torch.outer(t, freqs)
        self.register_buffer('rope_cos', torch.cos(angles), persistent=False)
        self.register_buffer('rope_sin', torch.sin(angles), persistent=False)

    def _apply_rope(self, x, pos):
        """Apply rotary position embedding at position pos to (B, H, S, head_dim)."""
        D2 = x.shape[-1] // 2
        cos = self.rope_cos[pos:pos + 1, :D2].to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin[pos:pos + 1, :D2].to(device=x.device, dtype=x.dtype)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        y = torch.empty_like(x)
        y[..., 0::2] = rot_even
        y[..., 1::2] = rot_odd
        return y

    def forward(self, x, pos, past_kv=None, use_cache=False):
        """Forward with RoPE on Q/K at position pos.

        Args:
            x: (B, 1, D) input token
            pos: scalar int, current position index (0..T-1)
            past_kv: optional (past_k, past_v) for KV-cache
            use_cache: if True, return (output, new_kv)
        """
        B, S, D = x.shape

        normed = self.norm1(x)
        q = self.q_proj(normed).view(B, S, self.heads, self.head_dim)
        k = self.k_proj(normed).view(B, S, self.kv_heads, self.head_dim)
        v = self.v_proj(normed).view(B, S, self.kv_heads, self.head_dim)

        # Apply RoPE to Q and K individually at current position
        q = self._apply_rope(q, pos).transpose(1, 2)  # (B, heads, 1, head_dim)
        k = self._apply_rope(k, pos).transpose(1, 2)  # (B, kv_heads, 1, head_dim)
        v = v.transpose(1, 2)                          # (B, kv_heads, 1, head_dim)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        if self.kv_heads < self.heads:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out_attn = torch.matmul(attn, v)
        out_attn = out_attn.transpose(1, 2).contiguous().view(B, S, D)
        out_attn = self.out_proj(out_attn)

        x = x + out_attn
        x = x + self.mlp(self.norm2(x))

        if use_cache:
            return x, new_kv
        return x


# ═══════════════════════════════════════════════════════════
#  RoPE Decoder — clean autoregressive backbone
# ═══════════════════════════════════════════════════════════

class RoPEDecoder(nn.Module):
    """Autoregressive decoder: z → KV-cache Transformer (with RoPE on Q/K) → tokens.

    z injected at EVERY step (not just step 0).
    Heads (color_head, mask_head) set externally by TokenPainter.

    Composition: softmax across tokens (sum=1, no background).
    """

    def __init__(self, z_dim=128, token_dim=512, num_tokens=8,
                 num_layers=4, heads=4, mlp_ratio=8.0, dropout=0.1,
                 use_input_proj=True, output_size=64, mask_temp=1.0,
                 num_classes=0, label_embed_dim=128):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.output_size = output_size
        self.mask_temp = mask_temp
        self.num_classes = num_classes

        # ── class conditioning ──
        if num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, label_embed_dim)
            z_in_dim = z_dim + label_embed_dim
        else:
            self.label_embed = None
            z_in_dim = z_dim

        # ── z → conditioning ──
        self.z_proj = nn.Linear(z_in_dim, token_dim, bias=True)

        # ── output → input projection (step 1..T-1) ──
        if use_input_proj:
            self.input_proj = nn.Sequential(
                nn.Linear(token_dim, token_dim * 2),
                nn.GELU(),
                nn.Linear(token_dim * 2, token_dim),
            )
        else:
            self.input_proj = nn.Identity()

        # ── Transformer blocks (RoPE is inside each block) ──
        max_pos = max(num_tokens * 8, 256)
        self.blocks = nn.ModuleList([
            KVCacheBlock(token_dim, heads, heads, mlp_ratio, dropout, max_pos=max_pos)
            for _ in range(num_layers)
        ])

        # ── Heads (set externally by TokenPainter) ──
        self.color_head = None
        self.mask_head = None

    # ── Forward ──
    def forward(self, z, labels=None):
        """Autoregressive forward: z injected at every step.

        Args:
            z: (B, z_dim) latent codes
            labels: (B,) integer class labels, or None for unconditional

        Returns: canvas (B,3,H,W), colors (B,T,3), masks (B,T,H,W)
        """
        B = z.shape[0]
        T = self.num_tokens
        H = W = self.output_size

        # Class conditioning: concat label embedding to z
        if labels is not None and self.label_embed is not None:
            label_emb = self.label_embed(labels)  # (B, label_embed_dim)
            z = torch.cat([z, label_emb], dim=1)

        z_cond = self.z_proj(z)  # (B, dim)
        past_kv = [None] * len(self.blocks)
        tokens_list = []

        for k in range(T):
            if k == 0:
                x = z_cond.unsqueeze(1)
            else:
                x = self.input_proj(tokens_list[-1]) + z_cond.unsqueeze(1)

            for i, blk in enumerate(self.blocks):
                x, new_kv = blk(x, pos=k, past_kv=past_kv[i], use_cache=True)
                past_kv[i] = new_kv

            tokens_list.append(x)

        tokens = torch.cat(tokens_list, dim=1)  # (B, T, dim)

        # ── Colors ──
        colors = self.color_head(tokens)  # (B, T, 3)
        c_full = colors.unsqueeze(-1).unsqueeze(-1)  # (B, T, 3, 1, 1)

        # ── Masks ──
        tokens_flat = tokens.reshape(B * T, self.token_dim)
        mask_logits = self.mask_head.forward_logits(tokens_flat).reshape(B, T, H, W)

        # Softmax across tokens: sum_t w_t = 1, no background needed
        masks = F.softmax(mask_logits / self.mask_temp, dim=1)  # (B, T, H, W)
        canvas = (masks.unsqueeze(2) * c_full).sum(dim=1)

        return canvas, colors, masks
