"""TokenPainter: autoregressive token-based image generation.

Usage:
    from TokenPainter import TokenPainter64, TokenPainter32

    model_64 = TokenPainter64(num_tokens=8)  # 64×64 output (MF, CelebA)
    model_32 = TokenPainter32(num_tokens=8)  # 32×32 output (CIFAR-10, MNIST)
"""

import torch.nn as nn

from .backbone import RoPEDecoder
from .heads import RGBHeadDeep, TP64MaskHead, TP64HeavyMaskHead, TP32MaskHead

_SHARED = dict(z_dim=128, token_dim=512, heads=4, mlp_ratio=8.0, dropout=0.0, num_layers=4)


class TokenPainter64(nn.Module):
    """TokenPainter for 64×64 images (MetFaces, CelebA).

    G = RoPEDecoder + RGBHeadDeep + TP64MaskHead (~90M params).
    """
    def __init__(self, num_tokens=8,                  token_dim=512, heads=4, rgb_hidden=512, mask_temp=1.0,
                 dropout=0.0, num_classes=0, label_embed_dim=128):
        super().__init__()
        shared = {**_SHARED, 'token_dim': token_dim, 'heads': heads, 'dropout': dropout,
                  'num_classes': num_classes, 'label_embed_dim': label_embed_dim}
        self.backbone = RoPEDecoder(
            num_tokens=num_tokens,
            use_input_proj=True, output_size=64,
            mask_temp=mask_temp, **shared,
        )
        self.backbone.color_head = RGBHeadDeep(token_dim=token_dim, hidden=rgb_hidden, use_ln=True)
        self.backbone.mask_head = TP64MaskHead(token_dim=token_dim)

    def forward(self, z, labels=None):
        return self.backbone(z, labels)


class TokenPainter64Heavy(nn.Module):
    """TokenPainter 64×64 with heavy mask head (base_ch=256, post_ch=96, deep_refine).

    G = RoPEDecoder + RGBHeadDeep + TP64HeavyMaskHead (~160M params).
    """
    def __init__(self, num_tokens=8,                  token_dim=512, heads=4, rgb_hidden=512, mask_temp=1.0,
                 dropout=0.0, num_classes=0, label_embed_dim=128):
        super().__init__()
        shared = {**_SHARED, 'token_dim': token_dim, 'heads': heads, 'dropout': dropout,
                  'num_classes': num_classes, 'label_embed_dim': label_embed_dim}
        self.backbone = RoPEDecoder(
            num_tokens=num_tokens,
            use_input_proj=True, output_size=64,
            mask_temp=mask_temp, **shared,
        )
        self.backbone.color_head = RGBHeadDeep(token_dim=token_dim, hidden=rgb_hidden, use_ln=True)
        self.backbone.mask_head = TP64HeavyMaskHead(token_dim=token_dim)

    def forward(self, z, labels=None):
        return self.backbone(z, labels)


class TokenPainter32(nn.Module):
    """TokenPainter for 32×32 images (CIFAR-10, MNIST, Omniglot).

    G = RoPEDecoder + RGBHeadDeep + TP32MaskHead (~76M params).
    """
    def __init__(self, num_tokens=8,                  token_dim=512, heads=4, rgb_hidden=512, mask_temp=1.0,
                 dropout=0.0, num_classes=0, label_embed_dim=128):
        super().__init__()
        shared = {**_SHARED, 'token_dim': token_dim, 'heads': heads, 'dropout': dropout,
                  'num_classes': num_classes, 'label_embed_dim': label_embed_dim}
        self.backbone = RoPEDecoder(
            num_tokens=num_tokens,
            use_input_proj=True, output_size=32,
            mask_temp=mask_temp, **shared,
        )
        self.backbone.color_head = RGBHeadDeep(token_dim=token_dim, hidden=rgb_hidden, use_ln=True)
        self.backbone.mask_head = TP32MaskHead(token_dim=token_dim)

    def forward(self, z, labels=None):
        return self.backbone(z, labels)
