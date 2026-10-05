"""SAGAN 共享层：自注意力、谱归一化卷积等基础模块。

   对齐官方 TensorFlow 实现 (brain-research/self-attention-gan)，
   转为 PyTorch 版本。

   === 谱归一化 (Spectral Normalization) ===

   定义：对权重矩阵 W，除以它的最大奇异值（谱范数）：
         W_sn = W / σ_max(W)
   谱范数是 W 的 2-范数，即 W·W^T 最大特征值的平方根。

   为什么需要？
     WGAN-GP 要求判别器是 1-Lipschitz 函数：||f(x)-f(y)|| ≤ ||x-y||。
     每层线性变换 Wx 的 Lipschitz 常数为 σ_max(W)，
     逐层谱归一化 → 每层 Lipschitz ≤ 1 → 整个网络 Lipschitz ≤ 1。

   怎么算？
     精确 SVD 是 O(n³)，用幂迭代法在线近似：
       u = W^T·v,  u = u / ||u||
       v = W·u,    v = v / ||v||
       σ_max ≈ u^T·W·v
     PyTorch 的 spectral_norm 在每次 forward 时用上一轮的 u,v 迭代一次。

   判别器和生成器都加 SN：
     D 需要 Lipschitz 约束保证 WGAN-GP 有效；
     G 加 SN 可以限制生成的突变，官方最终版 G 全部使用 SN。

   === 自注意力结构（与标准 Transformer 的不同） ===

   官方 SAGAN 的自注意力：
     Q (theta): 不降采样，保持全分辨率
     K (phi):   MaxPool(2×2) → 空间减为 1/4
     V (g):     MaxPool(2×2) → 空间减为 1/4，通道降为 C/2

   聚合后：out = softmax(Q·K^T)·V，reshape 后 1×1 卷积升回 C 通道。

   为什么 K/V 要降采样？
     注意力矩阵 Q·K^T 的形状是 (N, N/4)，比标准的 (N, N) 节省 4 倍计算量。
     像素艺术中邻近像素高度相关，2×2 池化几乎不损失有用信息。
   为什么不除 √d_k？
     同上——K 已经降采样，点积值在可控范围，不需要额外缩放。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ============================================================
# 工具函数
# ============================================================

def sn_conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Conv2d:
    """创建带谱归一化的卷积层（bias=True，对齐官方）。"""
    return spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=True))


def sn_conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    """1×1 谱归一化卷积（SAGAN 自注意力专用）。"""
    return spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=True))


def sn_linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    """谱归一化全连接层。"""
    return spectral_norm(nn.Linear(in_features, out_features, bias=bias))


# ============================================================
# 条件批归一化 (Conditional Batch Normalization, CBN)
# ============================================================

class ConditionalBatchNorm2d(nn.Module):
    """条件批归一化。完全对齐官方 ops.py ConditionalBatchNorm。

    CBN(x, y) = BN(x) × γ[y] + β[y]

    γ[y]: 第 y 类的缩放参数, 形状 (num_features,), 初始化为 1
    β[y]: 第 y 类的平移参数, 形状 (num_features,), 初始化为 0

    官方实现:
      self.gamma = tf.get_variable('gamma', [num_classes, num_features],
                                    initializer=tf.ones_initializer())
      self.beta  = tf.get_variable('beta',  [num_classes, num_features],
                                    initializer=tf.zeros_initializer())
      output = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Parameter(torch.ones(num_classes, num_features))
        self.beta  = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        g = self.gamma[labels]  # (B, C)
        b = self.beta[labels]   # (B, C)
        return self.bn(x) * g.unsqueeze(-1).unsqueeze(-1) + b.unsqueeze(-1).unsqueeze(-1)


# ============================================================
# 自注意力（对齐官方 non_local.py）
# ============================================================

class SelfAttention(nn.Module):
    """SAGAN 自注意力：Q 全分辨率，K/V 经 MaxPool 降采样。

    完整形状流（以 C=128, H=W=16 为例）：
      输入 x:                      [B, 128, 16, 16]
      theta (Q): 1×1 SNConv C→16  [B, 128, 16, 16] → [B, 16, 16, 16] → permute → [B, 256, 16]
      phi   (K): 1×1 SNConv C→16  [B, 128, 16, 16] → [B, 16, 16, 16] → MaxPool → [B, 16,  8,  8] → view → [B, 16, 64]
      g     (V): 1×1 SNConv C→64  [B, 128, 16, 16] → [B, 64, 16, 16] → MaxPool → [B, 64,  8,  8] → view → [B, 64, 64]
      attn = softmax(Q·K^T)       [B, 256, 16] @ [B, 16, 64]   → [B, 256, 64]
      out  = attn @ V             [B, 256, 64] @ [B, 64, 64]   → [B, 256, 64]
      out → view + 1×1 Conv 64→128 [B, 256, 64] → [B, 64, 16, 16] → [B, 128, 16, 16]
      输出: gamma×out + x          [B, 128, 16, 16]            形状不变，残差连接

    结构（官方 non_local.py 的 sn_non_local_block_sim）：
      theta (Q): 1×1 SN Conv, C → C//8, 不降采样
      phi   (K): 1×1 SN Conv, C → C//8, MaxPool(2×2) → 空间 /4
      g     (V): 1×1 SN Conv, C → C//2, MaxPool(2×2) → 空间 /4
      attn = softmax(theta @ phi^T)           # (B, N, N/4)
      out  = attn @ g                         # (B, N, C/2)
      out  = reshape + 1×1 SN Conv → C        # 投影回原通道
      output = gamma * out + x                # 残差连接

    对像素艺术的意义：
      MaxPool 在 2×2 邻域内保留最显著特征——像素角色中邻近像素颜色
      高度相关，池化不丢失有用信息。K/V 降采样后注意力计算量减 4 倍。

    gamma 从 0 开始可学习：
      初始化 gamma=0 → 输出=x → 退化为恒等映射。
      训练中 gamma 逐渐增大 → 先学局部特征，再逐步引入全局注意力。
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # Q (theta): 不降采样，每个位置都可以查询所有位置
        self.theta = sn_conv1x1(in_channels, in_channels // 8)

        # K (phi): MaxPool 降采样，减少被查询的 key 数量
        self.phi = sn_conv1x1(in_channels, in_channels // 8)

        # V (g): MaxPool 降采样 + 通道减半，减少聚合计算量
        self.g = sn_conv1x1(in_channels, in_channels // 2)

        # 最后的 1×1 卷积：把 C/2 投影回 C，才能与残差相加
        self.out_conv = sn_conv1x1(in_channels // 2, in_channels)

        # gamma: 从 0 开始，让网络自己决定何时引入全局注意力
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) 注意力增强后的特征图

        形状跟踪 (以常见 C=128, H=W=16 为例):
            theta: (B, 128, 16, 16) → 1×1 Conv → (B, 16, 16, 16) → view+permute → (B, 256, 16)
            phi:   (B, 128, 16, 16) → 1×1 Conv → (B, 16, 16, 16) → MaxPool2×2 → (B, 16, 8, 8) → view → (B, 16, 64)
            g:     (B, 128, 16, 16) → 1×1 Conv → (B, 64, 16, 16) → MaxPool2×2 → (B, 64, 8, 8) → view → (B, 64, 64)
            attn:   theta @ phi^T = (B, 256, 16) @ (B, 16, 64) → (B, 256, 64)   [N=256, N/4=64]
            out:    attn @ V = (B, 256, 64) @ (B, 64, 64) → (B, 256, 64) → view → (B, 64, 16, 16) → out_conv → (B, 128, 16, 16)
        """
        B, C, H, W = x.shape
        N = H * W                        # 全分辨率位置数 = Query 数量
        N_down = (H // 2) * (W // 2)     # 降采样后位置数 = Key/Value 数量

        # ===== Q (theta) — 全分辨率查询 =====
        # 1×1 SNConv: C → C//8, 不降采样
        # view + permute 将 (B, C//8, H, W) 展平为 (B, C//8, N) 再转置为 (B, N, C//8)
        # 每行对应一个空间位置对所有其他位置的查询向量
        theta = self.theta(x).view(B, C // 8, N).permute(0, 2, 1)  # (B, N, C//8)

        # ===== K (phi) — 降采样后的键 =====
        # 1×1 SNConv: C → C//8
        # F.max_pool2d( kernel_size=2, stride=2 ): 在 2×2 邻域内取最大值
        #   形状变化: (B, C//8, H, W) → (B, C//8, H/2, W/2)
        #   作用: 空间位置数从 N 减为 N/4, Q·K^T 矩阵从 (N, N) 降为 (N, N/4), 节省 4 倍计算
        #   合理性: 像素艺术中邻近像素高度相关, 2×2 池化几乎不丢失有用信息
        phi = F.max_pool2d(self.phi(x), kernel_size=2, stride=2, return_indices=True)[0]  # (B, C//8, H/2, W/2)
        phi = phi.view(B, C // 8, N_down)  # (B, C//8, N/4)

        # ===== V (g) — 降采样后的值 =====
        # 1×1 SNConv: C → C//2 (通道保留更多, 因为最终聚合输出直接来自 V)
        # MaxPool 同样 2×2, stride=2: H×W → H/2×W/2
        g = F.max_pool2d(self.g(x), kernel_size=2, stride=2, return_indices=True)[0]  # (B, C//2, H/2, W/2)
        g = g.view(B, C // 2, N_down)  # (B, C//2, N/4)

        # ===== 注意力矩阵 =====
        # theta @ phi: (B, N, C//8) @ (B, C//8, N/4) → (B, N, N/4)
        # 第 i 行 j 列 = 位置 i 对位置 j 的注意力分数 (规范化后)
        # 注意: 标准 Transformer 要除 √d_k 防止梯度消失, 但官方 SAGAN 不除, 实验证明显式更好
        attn = torch.softmax(torch.bmm(theta, phi), dim=-1)  # (B, N, N/4)

        # ===== 注意力聚合 =====
        # attn @ V: (B, N, N/4) @ (B, N/4, C//2) → (B, N, C//2)
        # 每个位置 i 的输出 = Σ_j attn[i,j] * V[j], 即所有 value 位置的加权平均
        out = torch.bmm(attn, g.permute(0, 2, 1))  # (B, N, C//2)

        # ===== 恢复空间结构 + 通道投影 =====
        # (B, N, C//2) → (B, C//2, H, W), 恢复 2D 空间排布
        out = out.view(B, C // 2, H, W)
        # 1×1 SNConv: C//2 → C, 匹配残差连接的通道数
        out = self.out_conv(out)  # (B, C, H, W)

        # gamma 从 0 开始可学习: 初始输出=x (恒等映射), 训练中 gamma 逐渐增大
        return self.gamma * out + x  # (B, C, H, W)
