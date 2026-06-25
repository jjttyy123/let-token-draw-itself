"""SAGAN 模型：对齐官方 brain-research/self-attention-gan 实现 (PyTorch 版)。

   Phase 1: 无条件 D + G
   Phase 2: 条件 D (Projection) + 条件 G (CBN)
"""

from .layers import SelfAttention, ConditionalBatchNorm2d, sn_conv, sn_conv1x1, sn_linear
from .discriminator import UnconditionalDiscriminator, ConditionalDiscriminator, DResBlock, DOptimizedBlock
from .generator import UnconditionalGenerator, ConditionalGenerator, GResBlock, CGResBlock
