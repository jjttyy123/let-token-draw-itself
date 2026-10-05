"""SAGAN 无条件判别器（对齐官方 TensorFlow 实现）。

   核心改动（相比初版）：
     - ResBlock 替代顺序 Conv+LeakyReLU
     - pre-activation: ReLU 在卷积之前
     - AvgPool 下采样替代 stride=2 Conv
     - 所有 Conv/Linear 加谱归一化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SelfAttention, sn_conv, sn_conv1x1, sn_linear


# ============================================================
# D 专用 ResBlock
# ============================================================

class DResBlock(nn.Module):
    """判别器残差块（对齐官方 block / optimized_block）。

    pre-activation 模式：ReLU → SNConv3×3 → ReLU → SNConv3×3 → +skip
    可选 AvgPool 下采样（官方用 2×2 avg_pool 减半分辨率）。
    skip 连接通过 1×1 SNConv 匹配通道 + AvgPool 匹配尺寸。
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True, use_sn: bool = True):
        super().__init__()
        self.downsample = downsample
        conv = sn_conv if use_sn else lambda ic, oc, k, s, p: nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p)
        conv1x1 = sn_conv1x1 if use_sn else lambda ic, oc: nn.Conv2d(ic, oc, kernel_size=1)

        self.conv1 = conv(in_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = conv(out_ch, out_ch, k=3, s=1, p=1)
        self.skip_conv = conv1x1(in_ch, out_ch) if in_ch != out_ch or downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            downsample=True  → (B, out_ch, H/2, W/2)   下采样后的特征图
            downsample=False → (B, out_ch, H,  W)      保持空间尺寸

        形状跟踪 (以 in_ch=64, out_ch=128, H=W=32, downsample=True 为例):
            x:     (B,  64, 32, 32)
            relu:  (B,  64, 32, 32)
            conv1: (B, 128, 32, 32)
            relu:  (B, 128, 32, 32)
            conv2: (B, 128, 32, 32)
            pool:  (B, 128, 16, 16)
            skip_conv: (B, 128, 32, 32)
            skip_pool: (B, 128, 16, 16)
            sum:   (B, 128, 16, 16)
        """
        # ========== 主路径 (pre-activation: 先激活后卷积) ==========
        h = F.relu(x)               # (B, in_ch, H, W)      pre-act: 先激活再卷积
        h = self.conv1(h)           # SNConv 3×3, s=1, p=1 → (B, out_ch, H, W)  保持空间
        h = F.relu(h)               # (B, out_ch, H, W)
        h = self.conv2(h)           # SNConv 3×3, s=1, p=1 → (B, out_ch, H, W)

        # F.avg_pool2d(主路径下采样)
        #   kernel_size=2, stride=2 → 每个 2×2 区域取平均值, 空间减半
        #   为什么不用 stride=2 Conv? 官方 SAGAN 实验表明 AvgPool 训练更稳定,
        #   不含可学习参数, 纯粹几何下采样, 能保留全局均值信息
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)  # (B, out_ch, H/2, W/2)

        # ========== 跳过连接 ==========
        # self.skip_conv: 当 in_ch != out_ch 或需要 downsample 时用 1×1 SNConv
        #   形状变化: (B, in_ch, H, W) → (B, out_ch, H, W)
        #   1×1 卷积只改变通道数, 不改变空间分辨率
        #   当 in_ch==out_ch 且 downsample=False 时, skip_conv 是 Identity, 直接传 x
        skip = self.skip_conv(x)    # (B, out_ch, H, W) 或 (B, in_ch, H, W)

        # skip 路径也必须下采样, 这样才能与主路径形状一致后逐元素相加
        if self.downsample:
            skip = F.avg_pool2d(skip, kernel_size=2, stride=2)  # (B, out_ch, H/2, W/2)

        return h + skip              # 残差相加, 形状由 downsample 决定


class DOptimizedBlock(nn.Module):
    """判别器第一个 block（对齐官方 optimized_block）。

    与 DResBlock 的区别：
      - 没有第一个 ReLU（pre-activation 不适用于 RGB 输入层）
      - 输入直接进卷积，然后 ReLU → Conv → downsample
      - skip: 直接 downsample + 1×1 conv
    """

    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        conv = sn_conv if use_sn else lambda ic, oc, k, s, p: nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p)
        conv1x1 = sn_conv1x1 if use_sn else lambda ic, oc: nn.Conv2d(ic, oc, kernel_size=1)

        self.conv1 = conv(in_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = conv(out_ch, out_ch, k=3, s=1, p=1)
        self.skip_conv = conv1x1(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W), 优化块输入 (通常为 RGB 图像 3×64×64)
        Returns:
            (B, out_ch, H/2, W/2) 经 AvgPool 下采样后的特征图

        形状跟踪 (以 in_ch=3, out_ch=64, H=W=64 为例):
            x:     (B,  3, 64, 64)
            conv1: (B, 64, 64, 64)   # 原始 RGB 值直接进入第一个卷积 (无 pre-act)
            relu:  (B, 64, 64, 64)
            conv2: (B, 64, 64, 64)
            pool:  (B, 64, 32, 32)   # AvgPool(2,2): 空间减半
            skip_pool: (B,  3, 32, 32)
            skip_conv: (B, 64, 32, 32)
            sum:   (B, 64, 32, 32)
        """
        # ========== 主路径 (无第一个 ReLU — 与 DResBlock 不同) ==========
        # DResBlock 先 ReLU 再卷积 (pre-activation),
        # 但输入是原始 RGB 像素值, 预处理 ReLU 会丢失负值信息 (如 [-1,1] 中的负数),
        # 所以优化块先做卷积, 再 ReLU, 再卷积
        h = self.conv1(x)           # SNConv 3×3, p=1, s=1 → (B, out_ch, H, W)  保持空间尺寸
        h = F.relu(h)               # (B, out_ch, H, W)
        h = self.conv2(h)           # SNConv 3×3, p=1, s=1 → (B, out_ch, H, W)
        # F.avg_pool2d(kernel_size=2, stride=2):
        #   对特征图做 2×2 平均池化, 空间尺寸从 H×W 减为 H/2×W/2
        #   与 stride=2 的卷积相比:
        #     - 无可学习参数 → 更稳定, 不易过拟合
        #     - 保留邻域均值 → 适合判别器需要提取整体特征的需求
        h = F.avg_pool2d(h, kernel_size=2, stride=2)  # (B, out_ch, H/2, W/2)

        # ========== 跳过连接 (先 pool 再 1×1 conv) ==========
        # 注意: 这里先对原始输入 x 做 AvgPool 降采样, 再用 1×1 Conv 升通道
        # 而 DResBlock 是先 1×1 Conv 再 avg_pool (当 downsample=True 时)
        # 顺序不同但结果等价, 因为 avg_pool 和 1×1 conv 是可交换的线性操作
        skip = F.avg_pool2d(x, kernel_size=2, stride=2)  # (B, in_ch, H/2, W/2)
        skip = self.skip_conv(skip)  # 1×1 SNConv: in_ch→out_ch → (B, out_ch, H/2, W/2)

        return h + skip              # (B, out_ch, H/2, W/2)


# ============================================================
# 无条件判别器
# ============================================================

class UnconditionalDiscriminator(nn.Module):
    """SAGAN 无条件判别器：ResBlock + Self-Attention → 单标量。

    结构（5 个 block，df_dim=64 时完整形状流）：
      [B,   3,  64, 64]     输入 RGB 图像
      → [DOptimizedBlock ↓2]  Conv→ReLU→Conv→AvgPool + skip(Pool→Conv)
                              [B,  64,  32, 32]     3→64ch, 64→32
      → [DResBlock ↓2]        ReLU→SNConv→ReLU→SNConv→AvgPool
                              [B, 128,  16, 16]     64→128ch, 32→16
      → [SelfAttention]       Q全分辨率, K/V MaxPool降采样
                              [B, 128,  16, 16]     空间256位置, 形状不变
      → [DResBlock ↓2]        ReLU→SNConv→ReLU→SNConv→AvgPool
                              [B, 256,   8,  8]     128→256ch, 16→8
      → [DResBlock ↓2]        ReLU→SNConv→ReLU→SNConv→AvgPool
                              [B, 512,   4,  4]     256→512ch, 8→4
      → [DResBlock  =]        ReLU→SNConv→ReLU→SNConv (无下采样)
                              [B, 512,   4,  4]     512→512ch, 4→4
      → ReLU → sum(H,W) → Linear 512→1
                              [B,   1]              WGAN-GP 判别分数

    参数量 ~1M（df_dim=64 时）。
    """

    def __init__(self, in_channels: int = 3, df_dim: int = 64, use_sn: bool = True):
        """
        Args:
            in_channels: 输入通道数（RGB=3）
            df_dim:     基础通道数，每层翻倍
            use_sn:     是否使用 SpectralNorm（默认 True）
        """
        super().__init__()

        self.block1 = DOptimizedBlock(in_channels, df_dim, use_sn=use_sn)
        self.block2 = DResBlock(df_dim, df_dim * 2, downsample=True, use_sn=use_sn)
        self.attn = SelfAttention(df_dim * 2)
        self.block3 = DResBlock(df_dim * 2, df_dim * 4, downsample=True, use_sn=use_sn)
        self.block4 = DResBlock(df_dim * 4, df_dim * 8, downsample=True, use_sn=use_sn)
        self.block5 = DResBlock(df_dim * 8, df_dim * 8, downsample=False, use_sn=use_sn)
        self.fc = sn_linear(df_dim * 8, 1) if use_sn else nn.Linear(df_dim * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 64, 64) RGB 图像，值域 [-1, 1]
        Returns:
            (B, 1) WGAN-GP 实数判别分数

        完整形状跟踪:
            x:       (B,   3, 64, 64)   输入 RGB
            block1:  (B,  64, 32, 32)   DOptimizedBlock ↓2,   3→ 64
            block2:  (B, 128, 16, 16)   DResBlock ↓2,        64→128
            attn:    (B, 128, 16, 16)   SelfAttention, 16×16=256 位置, 形状不变
            block3:  (B, 256,  8,  8)   DResBlock ↓2,       128→256
            block4:  (B, 512,  4,  4)   DResBlock ↓2,       256→512
            block5:  (B, 512,  4,  4)   DResBlock 不变,      512→512 (保持 4×4)
            relu:    (B, 512,  4,  4)
            sum:     (B, 512)           空间所有像素求和
            fc:      (B,   1)           线性投影到实数, WGAN-GP 判别分数
        """
        # ===== 编码器: 逐步下采样 (↓2), 通道数逐块翻倍 =====
        # block1 是 DOptimizedBlock: 没有 pre-act ReLU, 直接从 RGB 进卷积
        x = self.block1(x)    # (B,  64, 32, 32)   64×64 → 32×32,  3→ 64 ch
        # block2~5 是 DResBlock: pre-activation (ReLU → Conv → ReLU → Conv)
        x = self.block2(x)    # (B, 128, 16, 16)   32×32 → 16×16, 64→128 ch
        # Self-Attention: 空间 16×16 (256 个位置), Q 全分辨率, K/V MaxPool(2×2)
        # 引入全局感受野, 让判别器能捕捉远距离像素关系
        x = self.attn(x)      # (B, 128, 16, 16)   形状不变化
        x = self.block3(x)    # (B, 256,  8,  8)   16×16 → 8×8,  128→256 ch
        x = self.block4(x)    # (B, 512,  4,  4)   8×8   → 4×4,  256→512 ch
        # 最后一个 block 不下采样: 4×4 已经很小, 保留空间信息
        x = self.block5(x)    # (B, 512,  4,  4)   空间不变, 512→512 ch

        # ===== 输出头: 映射到标量 =====
        x = F.relu(x)                    # (B, 512, 4, 4)
        # 空间求和: 将 4×4=16 个空间位置的值累加
        # 注意: 官方用 tf.reduce_sum (求和), 不是 reduce_mean (平均)
        # 所以输出值尺度与特征图大小 (4×4=16) 成正比
        x = x.sum(dim=[2, 3])            # (B, 512)  将 H, W 两维求和
        x = self.fc(x)                   # (B, 1)  SNLinear: 512→1
        return x


# ============================================================
# 条件判别器 (Phase 2) — Projection Discriminator
# ============================================================

class ConditionalDiscriminator(nn.Module):
    """SAGAN 条件判别器：Projection Discriminator (Miyato & Koyama, ICLR 2018)。

    D(x, y) = emb(y)^T · h + fc(h)
              └── 条件项 ──┘   └─ 无条件项 ┘

    Backbone 与 UnconditionalDiscriminator 完全一致，仅在输出头增加
    label embedding 做内积投影。

    权重兼容:
      ckpt = torch.load('phase1_ckpt.pt')
      D_cond.load_state_dict(ckpt['D'], strict=False)
      # label_embed 不在 ckpt 中 → 零初始化
      # backbone + fc 完美匹配 → D_cond(x,·) = D_uncond(x)
    """

    def __init__(self, in_channels: int = 3, df_dim: int = 64, num_classes: int = 0,
                 use_sn: bool = True):
        super().__init__()

        # Backbone — 与 UnconditionalDiscriminator 完全一致
        self.block1 = DOptimizedBlock(in_channels, df_dim, use_sn=use_sn)
        self.block2 = DResBlock(df_dim, df_dim * 2, downsample=True, use_sn=use_sn)
        self.attn   = SelfAttention(df_dim * 2)
        self.block3 = DResBlock(df_dim * 2, df_dim * 4, downsample=True, use_sn=use_sn)
        self.block4 = DResBlock(df_dim * 4, df_dim * 8, downsample=True, use_sn=use_sn)
        self.block5 = DResBlock(df_dim * 8, df_dim * 8, downsample=False, use_sn=use_sn)

        # 无条件输出头
        self.fc = nn.Linear(df_dim * 8, 1) if not use_sn else sn_linear(df_dim * 8, 1)

        # 条件投影（仅 num_classes > 0 时创建）
        if num_classes > 0:
            self.label_embed = (sn_linear(num_classes, df_dim * 8, bias=False) if use_sn
                                else nn.Linear(num_classes, df_dim * 8, bias=False))
        else:
            self.label_embed = None

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.attn(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.relu(h)
        h = h.sum(dim=[2, 3])                        # (B, df_dim*8=512)

        out = self.fc(h)                              # 无条件项 ψ(φ(x))

        # Projection: D(x, y) = emb(y)^T h + fc(h)   (Miyato & Koyama 2018)
        if self.label_embed is not None and labels is not None:
            y_onehot = F.one_hot(labels, self.label_embed.in_features).float()
            h_labels = self.label_embed(y_onehot)          # emb(y)
            out += (h * h_labels).sum(dim=1, keepdim=True) # emb(y)^T φ(x) 内积
        return out
