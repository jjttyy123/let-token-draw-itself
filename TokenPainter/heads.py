"""Mask Head and RGB Head for TokenPainter.

TP64MaskHead: FC→pre-conv→PixelShuffle, 64×64 output (MF, CelebA)
TP32MaskHead: FC→pre-conv→PixelShuffle, 32×32 output (CIFAR-10, MNIST)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  RGB Head
# ═══════════════════════════════════════════════════════════

class RGBHeadDeep(nn.Module):
    """Residual MLP: token → RGB [0,1], 3-layer with skip + LayerNorm."""
    def __init__(self, token_dim=512, hidden=512, act='gelu', use_ln=True):
        super().__init__()
        self.fc1 = nn.Linear(token_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 3)
        self.proj = nn.Linear(token_dim, hidden) if token_dim != hidden else nn.Identity()
        self.act = nn.GELU() if act == 'gelu' else nn.SiLU()
        self.use_ln = use_ln
        if use_ln:
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)

    def forward(self, tokens):
        h = self.fc1(tokens)
        if self.use_ln: h = self.ln1(h)
        h = self.act(h)
        h = self.fc2(h) + self.proj(tokens)
        if self.use_ln: h = self.ln2(h)
        h = self.act(h)
        return torch.sigmoid(self.fc_out(h))


# ═══════════════════════════════════════════════════════════
#  BigFC + PixelShuffle Mask Head (shared implementation)
# ═══════════════════════════════════════════════════════════

class _BigFCMaskHeadPS(nn.Module):
    """Big FC → pre-conv → PixelShuffle → refine. Internal implementation."""
    def __init__(self, token_dim=512, base_ch=128, init_size=32, target_size=64,
                 post_ch=64, groups=1, pre2_k1=False, num_pre=2, extra_refine=False,
                 deep_refine=False):
        super().__init__()
        self.init_size = init_size
        self.post_ch = post_ch
        self.fc = nn.Linear(token_dim, base_ch * init_size * init_size)

        pre_out = post_ch * 4  # channels before pixel_shuffle: post_ch × r²
        pre2_k = 1 if pre2_k1 else 3
        pre2_p = 0 if pre2_k1 else 1
        self.num_pre = num_pre
        if num_pre == 1:
            self.pre1 = nn.Conv2d(base_ch, pre_out, 3, 1, 1, groups=groups)
        else:
            self.pre1 = nn.Conv2d(base_ch, base_ch, 3, 1, 1, groups=groups)
            self.pre2 = nn.Conv2d(base_ch, pre_out, pre2_k, 1, pre2_p, groups=groups)

        self.extra_refine = extra_refine
        if extra_refine:
            self.refine0 = nn.Conv2d(post_ch, post_ch, 3, 1, 1)
        self.refine1 = nn.Conv2d(post_ch, post_ch // 2, 3, 1, 1)
        self.refine2 = nn.Conv2d(post_ch // 2, post_ch // 4, 3, 1, 1)
        self.deep_refine = deep_refine
        if deep_refine:
            self.refine3 = nn.Conv2d(post_ch // 4, post_ch // 8, 3, 1, 1)
            self.final = nn.Conv2d(post_ch // 8, 1, 3, 1, 1)
        else:
            self.final = nn.Conv2d(post_ch // 4, 1, 3, 1, 1)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward_logits(self, x):
        B = x.shape[0]
        h = self.act(self.fc(x)).view(B, -1, self.init_size, self.init_size)
        h = self.act(self.pre1(h))
        if self.num_pre == 2:
            h = self.act(self.pre2(h))
        h = F.pixel_shuffle(h, 2)
        if self.extra_refine:
            h = self.act(self.refine0(h)) + h
        h = self.act(self.refine1(h))
        h = self.act(self.refine2(h))
        if self.deep_refine:
            h = self.act(self.refine3(h))
        return self.final(h).squeeze(1)

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))


# ═══════════════════════════════════════════════════════════
#  Public Mask Heads: one per model
# ═══════════════════════════════════════════════════════════

class TP64MaskHead(_BigFCMaskHeadPS):
    """Mask head for TP64 (64×64). FC(512→128×32²)=67M + pre-conv (groups=4) + PixelShuffle."""
    def __init__(self, token_dim=512):
        super().__init__(token_dim=token_dim, base_ch=128, init_size=32,
                         target_size=64, post_ch=64, num_pre=2, groups=4, extra_refine=False)


class TP64HeavyMaskHead(_BigFCMaskHeadPS):
    """Heavy mask head for TP64 (64×64). FC(512→256×32²)=134M + deep refinement 96→48→24→12→1."""
    def __init__(self, token_dim=512):
        super().__init__(token_dim=token_dim, base_ch=256, init_size=32,
                         target_size=64, post_ch=96, num_pre=2, groups=4,
                         deep_refine=True)


class TP32MaskHead(_BigFCMaskHeadPS):
    """Mask head for TP32 (32×32). FC(512→384×16²)=50M + pre-conv + PixelShuffle."""
    def __init__(self, token_dim=512):
        super().__init__(token_dim=token_dim, base_ch=384, init_size=16,
                         target_size=32, post_ch=96, num_pre=2)
