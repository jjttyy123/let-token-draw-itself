"""SAGAN 无条件生成器。对齐官方 TensorFlow 实现。

   z → SNLinear → 4×4 → ResBlock ↑2 ×4 → Self-Attention → 64×64 RGB
   输出连续 RGB (tanh 到 [-1,1])，不加调色板限制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SelfAttention, ConditionalBatchNorm2d, sn_conv, sn_conv1x1, sn_linear


class GResBlock(nn.Module):
    """生成器残差块: BN → ReLU → Upsample(nearest) → SNConv3×3 → BN → ReLU → SNConv3×3 + skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_ch)
        self.conv1 = sn_conv(in_ch, out_ch, k=3, s=1, p=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = sn_conv(out_ch, out_ch, k=3, s=1, p=1)
        self.skip_conv = sn_conv1x1(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(F.interpolate(x, scale_factor=2, mode='nearest'))
        h = self.bn0(x)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return h + skip


class UnconditionalGenerator(nn.Module):
    """SAGAN 无条件生成器（对齐官方 128×128 → 截取前 4 block 适配 64×64）。

    通道严格对齐官方:
      init_ch = gf_dim * 16 = 1024
      block1: 1024 → 1024  (4→8, 第一 block 不降通道)
      block2: 1024 → 512   (8→16)
      block3: 512  → 256   (16→32)
      block4: 256  → 128   (32→64)
      output: BN→ReLU→SNConv(128,3)→Tanh → [B,3,64,64]

    参数量 ~9M。
    """

    def __init__(self, z_dim: int = 128, gf_dim: int = 64):
        super().__init__()
        init_ch = gf_dim * 16  # 1024，对齐官方

        self.fc = sn_linear(z_dim, init_ch * 4 * 4)          # (B, 1024*16)

        self.block1 = GResBlock(init_ch,       init_ch)       # 1024→1024,  4→8
        self.block2 = GResBlock(init_ch,       gf_dim * 8)    # 1024→512,   8→16
        self.attn   = SelfAttention(gf_dim * 8)                # 16×16
        self.block3 = GResBlock(gf_dim * 8,    gf_dim * 4)    # 512→256,   16→32
        self.block4 = GResBlock(gf_dim * 4,    gf_dim * 2)    # 256→128,   32→64

        self.out_bn   = nn.BatchNorm2d(gf_dim * 2)
        self.out_conv = sn_conv(gf_dim * 2, 3, k=3, s=1, p=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.fc(z).view(B, -1, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out_bn(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return torch.tanh(x)


# ============================================================
# 条件生成器 (Phase 2)
# ============================================================

class CGResBlock(nn.Module):
    """条件生成器残差块：CBN→ReLU→Upsample→SNConv→CBN→ReLU→SNConv + skip。

    与 GResBlock 结构完全一致，仅 BN → CBN。所有 Conv 带 SN。
    CBN 的 γ/β 初始化为零，加载无条件权重后 G(z,·) ≈ G_uncond(z)。
    """

    def __init__(self, in_ch: int, out_ch: int, num_classes: int):
        super().__init__()
        self.cbn0 = ConditionalBatchNorm2d(in_ch, num_classes)
        self.conv1 = sn_conv(in_ch, out_ch, k=3)
        self.cbn1 = ConditionalBatchNorm2d(out_ch, num_classes)
        self.conv2 = sn_conv(out_ch, out_ch, k=3)
        self.skip_conv = sn_conv1x1(in_ch, out_ch)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(F.interpolate(x, scale_factor=2, mode='nearest'))
        h = self.cbn0(x, labels)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.cbn1(h, labels)
        h = F.relu(h)
        h = self.conv2(h)
        return h + skip


class ConditionalGenerator(nn.Module):
    """SAGAN 条件生成器。Backbone 复用无条件版本，BN → CBN。

    架构（与 UnconditionalGenerator 完全一致，仅 BN→CBN）:
      z → SNLinear → 4×4 → CGResBlock ↑2 ×4 → Self-Attention → 64×64 RGB

    权重兼容:
      ckpt = torch.load('phase1_ckpt.pt')
      G_cond.load_state_dict(ckpt['G'], strict=False)
      # CBN 的 embed 不在 ckpt 中 → 随机初始化 (已置零)
      # backbone 权重完美匹配 → 加载后 G_cond(z,·) ≈ G_uncond(z)
    """

    def __init__(self, z_dim: int = 128, gf_dim: int = 64, num_classes: int = 4):
        super().__init__()
        init_ch = gf_dim * 16

        self.fc = sn_linear(z_dim, init_ch * 4 * 4)

        self.block1 = CGResBlock(init_ch, init_ch, num_classes)        # 1024→1024, 4→8
        self.block2 = CGResBlock(init_ch, gf_dim * 8, num_classes)     # 1024→512,  8→16
        self.attn   = SelfAttention(gf_dim * 8)
        self.block3 = CGResBlock(gf_dim * 8, gf_dim * 4, num_classes)  # 512→256,  16→32
        self.block4 = CGResBlock(gf_dim * 4, gf_dim * 2, num_classes)  # 256→128,  32→64

        self.out_bn   = nn.BatchNorm2d(gf_dim * 2)
        self.out_conv = sn_conv(gf_dim * 2, 3, k=3)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.fc(z).view(B, -1, 4, 4)
        x = self.block1(x, labels)
        x = self.block2(x, labels)
        x = self.attn(x)
        x = self.block3(x, labels)
        x = self.block4(x, labels)
        x = self.out_bn(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return torch.tanh(x)
