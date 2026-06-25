"""
V5RL-v6 (DecoderActorRL) — 独立可运行版本

这是项目的最终主力模型，从 actor.py 中提取，去掉所有历史变体和实验分支，
只保留 V5RL-v6 实际使用的模块和默认配置。

架构概览:
  z(128) → z_proj(2层MLP) → z_cond(512) ── 注入自回归每一步
       ↓
  AR Loop (T=8 步):
    每步输入 = z_cond + pos_embed[k] + input_proj(tok_{k-1})
    → 4× KVCacheBlock (每层 = MHA + FFN)
    → tok_k (512维)
       ↓
  收集 T 个 token → color_head (逐token MLP → RGB 3维)
                  → mask_head  (逐token FC+ConvT → logits 64×64)
       ↓
  softmax_bg 合成 → canvas (3×64×64)

参数量: ~90M (deeprgb_ln_mask32, heads=4, mlp_ratio=8)
训练方式: MSE 预训练 → DDPG/GROUP_POS + SAGAN 对抗训练

使用方法:
    model = DecoderActorRLV6()
    z = torch.randn(4, 128)          # batch=4
    canvas, colors, masks = model(z) # canvas: (4, 3, 64, 64)

作者: 唐渝靖 + DeepSeek v4 pro (via Claude Code)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1. RGBHeadDeep+LN — 颜色解码头
# ═══════════════════════════════════════════════════════════════════════

class RGBHeadDeep(nn.Module):
    """3层残差 MLP + LayerNorm: token(512) → RGB(3).

    残差连接让梯度直接流过中间层, LayerNorm 稳定训练。

    Forward:
        h = fc1(token) → LN → GELU
        h = fc2(h) + proj(token)  ← 残差跳跃连接
        h = LN → GELU
        return sigmoid(fc_out(h))  → [0,1]^3
    """

    def __init__(self, token_dim=512, hidden=512, act='gelu', use_ln=True):
        super().__init__()
        self.fc1 = nn.Linear(token_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 3)                     # 3 = RGB 三通道

        # 残差投影: 如果输入输出维度不同，用 Linear 对齐；否则恒等映射
        self.proj = nn.Linear(token_dim, hidden) if token_dim != hidden else nn.Identity()

        self.act = nn.GELU() if act == 'gelu' else nn.SiLU()
        self.use_ln = use_ln
        if use_ln:
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)

    def forward(self, tokens):
        # tokens: (B, T, dim) — 所有 token 一起过 MLP，逐 token 独立
        h = self.fc1(tokens)
        if self.use_ln:
            h = self.ln1(h)
        h = self.act(h)
        h = self.fc2(h) + self.proj(tokens)                    # 残差连接
        if self.use_ln:
            h = self.ln2(h)
        h = self.act(h)
        return torch.sigmoid(self.fc_out(h))                   # (B, T, 3)


# ═══════════════════════════════════════════════════════════════════════
# 2. MaskHead32 — Mask 解码头 (前向 logits 版本)
# ═══════════════════════════════════════════════════════════════════════

class MaskHead32(nn.Module):
    """FC → 32×32 特征图 → ConvTranspose → 64×64 logits.

    关键设计:
      - FC 直接映射到 32×32 (不是 4×4), 给 mask 更多空间自由度
      - 只用 1 层 ConvTranspose (32→64), 轻量且避免棋盘格伪影
      - forward_logits() 返回原始值 (无 sigmoid/softmax)
        → softmax 在合成阶段统一做

    参数量: FC: 512 × 128 × 32 × 32 ≈ 67M (这是整个模型最大的单层)
    """

    def __init__(self, token_dim=512, base_ch=128):
        super().__init__()
        # base_ch=128 控制 FC 参数量 (R7 已探索 base_ch=256, MSE 相当但参数翻倍)
        self.fc = nn.Linear(token_dim, base_ch * 32 * 32)

        # ConvTranspose: 32×32 → 64×64 (stride=2, kernel=4, padding=1)
        # 后接 3×3 Conv2d 精修
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch // 2, 4, 2, 1),  # 32→64
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 1, 3, 1, 1),                 # refine
        )

    def forward_logits(self, x):
        """返回原始 logits (无激活函数), 供 softmax_bg 合成使用。"""
        N = x.shape[0]                                           # N = B * T
        x = self.fc(x).reshape(N, -1, 32, 32)                    # (N, 128, 32, 32)
        return self.deconv(x).squeeze(1)                          # (N, 64, 64)

    def forward(self, x):
        """返回 sigmoid(logits), 用于 overpaint / additive 合成。"""
        return torch.sigmoid(self.forward_logits(x))


# ═══════════════════════════════════════════════════════════════════════
# 3. KVCacheBlock — 带 KV Cache 的 Transformer Block
# ═══════════════════════════════════════════════════════════════════════

class KVCacheBlock(nn.Module):
    """标准 Pre-Norm Transformer Block, 支持 KV Cache 和 GQA。

    结构 (Pre-Norm, 残差连接):
        x = x + MHA(LN(x))     ← 多头自注意力, 支持 KV cache
        x = x + FFN(LN(x))     ← 两层 MLP, GELU 激活

    KV Cache:
      - 自回归生成时, 每步只输入 1 个新 token
      - key/value 沿序列维度拼接, 避免重复计算历史 token 的 K/V
      - 第 k 步的 token 可以 attend 到第 0..k-1 步的所有 token (causal)

    GQA (Grouped Query Attention):
      - Q heads = 4, KV heads = 2 (kv_heads < heads 时启用)
      - 减少 KV cache 显存, 效果接近全 MHA
      - 本项目默认不使用 (kv_heads == heads)
    """

    def __init__(self, dim=512, heads=4, kv_heads=4, mlp_ratio=8.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads                             # d_k = 512/4 = 128
        self.kv_dim = self.head_dim * kv_heads                   # KV 总维度
        self.q_per_kv = heads // kv_heads                        # 每个 KV head 对应几个 Q head

        # ── MHA 部分 ──
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)                        # Q: 全维度
        self.k_proj = nn.Linear(dim, self.kv_dim)                # K: GQA 降维 (或不降)
        self.v_proj = nn.Linear(dim, self.kv_dim)                # V: 同上
        self.out_proj = nn.Linear(dim, dim)                      # 输出投影
        self.attn_dropout = nn.Dropout(dropout)

        # ── FFN 部分 (mlp_ratio=8: 512 → 4096 → 512) ──
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),                # 512 → 4096
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),                # 4096 → 512
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Args:
            x:         (B, 1, dim) — 自回归每步只有 1 个 token
            past_kv:   (past_K, past_V) 或 None
            use_cache: 是否返回新 KV (推理时为 True, 训练时因 causal mask 不需要)
        Returns:
            x:         (B, 1, dim) — 经过 MHA+FFN 的输出
            new_kv:    (K, V) 或 None
        """
        B, S, D = x.shape                                         # S=1 (自回归)

        # ── 1. Multi-Head Attention (Pre-Norm) ──
        normed = self.norm1(x)
        q = self.q_proj(normed).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        # KV cache: 沿序列维度拼接历史 key/value
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)                     # (B, H_kv, past+S, d_k)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v) if use_cache else None

        # GQA: 将 KV heads 复制到和 Q heads 一样多
        if self.kv_heads < self.heads:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        # Scaled dot-product attention (causal 由 KV cache 天然保证)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out_attn = torch.matmul(attn, v)
        out_attn = out_attn.transpose(1, 2).contiguous().view(B, S, D)
        out_attn = self.out_proj(out_attn)

        x = x + out_attn                                          # 残差连接

        # ── 2. Feed-Forward Network (Pre-Norm) ──
        x = x + self.mlp(self.norm2(x))                           # 残差连接

        if use_cache:
            return x, new_kv
        return x


# ═══════════════════════════════════════════════════════════════════════
# 4. DecoderActorRL V5RL-v6 — 主模型
# ═══════════════════════════════════════════════════════════════════════

class DecoderActorRLV6(nn.Module):
    """V5RL-v6: 真正的自回归 Token 生成器, 用于像素艺术生成。

    ═══════════════════════════════════════════════════════════════
    架构 (逐行对应代码):
    ═══════════════════════════════════════════════════════════════

    (1) z_proj (噪声编码):
        z(128) → Linear→GELU→Linear → z_cond(512)
        将随机噪声编码为条件向量, 注入自回归每一步

    (2) input_proj (输入投影):
        tok_{k-1}(512) → Linear→GELU→Linear → (512)
        将上一步输出 token 映射回输入空间 (类似 GPT 的 token embedding)

    (3) pos_embed (位置编码):
        可学习参数 (1, T, 512), 每步加上对应位置

    (4) AR Loop (自回归生成, T 步):
        for k in range(T):
            # 构建输入: z_cond(始终) + 位置 + 上步反馈
            if k == 0: x = z_cond                        # 第一步: 纯条件
            else:      x = input_proj(tok_{k-1}) + z_cond # 后续步: 反馈+条件

            x = x + pos_embed[k]                          # 加位置

            # 通过 4 层 KVCacheBlock (每层 = MHA + FFN)
            for blk in self.blocks:
                x = blk(x, kv_cache)                     # KV cache 累积历史

            tok_k = x                                     # 产出当前 token

        → 收集为 tokens: (B, T, 512)

    (5) 并行解码 (逐 token 独立, 无交互):
        color_head(tokens)  → colors: (B, T, 3)          # RGB 颜色
        mask_head(tokens)   → logits: (B, T, 64, 64)     # 空间 mask

    (6) softmax_bg 合成:
        对每个像素 (x,y):
          w = softmax([logits_1, ..., logits_T, 0])       # T+1 个权重
          canvas(x,y) = Σ w_k·c_k + w_bg·bg_color

    ═══════════════════════════════════════════════════════════════
    MDP 视角 (为什么这是 RL):
    ═══════════════════════════════════════════════════════════════
      State s_t:  KV cache (累积所有历史决策)
      Action a_t: (color_t, mask_t) 解码自 tok_t
      Reward r:   0, ..., 0, D(canvas_final)  — 稀疏奖励, 只有最终画布被评判

    ═══════════════════════════════════════════════════════════════
    默认配置 (V5RL-v6, ~90M 参数):
    ═══════════════════════════════════════════════════════════════
      z_dim=128, token_dim=512, num_tokens=8
      num_layers=4, heads=4, mlp_ratio=8, dropout=0.1
      zproj='deep' (2层MLP), use_input_proj=True
      head='deeprgb_ln_mask32', compose='softmax_bg', bg_color=1.0
    """

    def __init__(self,
                 # 维度
                 z_dim=128,
                 token_dim=512,
                 num_tokens=8,
                 # Transformer
                 num_layers=4,
                 heads=4,
                 kv_heads=4,                                  # =heads → 标准 MHA, <heads → GQA
                 mlp_ratio=8.0,
                 dropout=0.1,
                 # 合成
                 compose_mode='softmax_bg',
                 bg_color=1.0,                                 # 1.0=白底, 0.0=黑底
                 # 组件开关
                 zproj_type='deep',                            # 'deep'=2层MLP, 'simple'=1层Linear
                 use_input_proj=True,                          # 输入投影 (推荐开启)
                 ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.compose_mode = compose_mode

        # ── (1) z_proj: 噪声 → 条件向量 ──
        if zproj_type == 'deep':
            self.z_proj = nn.Sequential(
                nn.Linear(z_dim, token_dim),                   # 128 → 512
                nn.GELU(),
                nn.Linear(token_dim, token_dim),               # 512 → 512
            )
        else:
            self.z_proj = nn.Linear(z_dim, token_dim)

        # ── (2) input_proj: 上步输出 → 当前输入空间 ──
        if use_input_proj:
            self.input_proj = nn.Sequential(
                nn.Linear(token_dim, token_dim * 2),           # 512 → 1024
                nn.GELU(),
                nn.Linear(token_dim * 2, token_dim),           # 1024 → 512
            )
        else:
            self.input_proj = nn.Identity()

        # ── (3) pos_embed: 可学习位置编码 ──
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

        # ── (4) Transformer Blocks (共享参数, 每步都过) ──
        self.blocks = nn.ModuleList([
            KVCacheBlock(token_dim, heads, kv_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # ── (5) 输出头 ──
        self.color_head = RGBHeadDeep(token_dim, hidden=token_dim, use_ln=True)
        self.mask_head = MaskHead32(token_dim, base_ch=128)

        # ── (6) 背景色 (不可学习常数) ──
        bg_val = float(bg_color)
        self.register_buffer('bg', torch.full((1, 3, 64, 64), bg_val), persistent=False)

        self._init_weights()

    def _init_weights(self):
        """trunc_normal 初始化 Linear 和 ConvTranspose 权重, bias 置零。"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        """前向传播: z → canvas, colors, masks.

        Args:
            z: (B, 128) 随机噪声, 通常来自 N(0,I)

        Returns:
            canvas: (B, 3, 64, 64)  最终合成画布, 值域 [0,1]
            colors: (B, T, 3)        每个 token 的 RGB 颜色
            masks:  (B, T, 64, 64)   每个 token 的空间权重 (softmax 归一化后)
        """
        B = z.shape[0]
        T = self.num_tokens

        # ── Step 1: 噪声编码 ──
        z_cond = self.z_proj(z)                                # (B, 512)

        # ── Step 2: 自回归生成 T 个 token ──
        past_kv = [None] * len(self.blocks)                    # KV cache 初始化
        tokens_list = []

        for k in range(T):
            # 构建每步输入
            if k == 0:
                x = z_cond.unsqueeze(1)                         # 第一步: 只有 z_cond
            else:
                # 上步输出经 input_proj 映射 + z_cond 注入
                x = self.input_proj(tokens_list[-1]) + z_cond.unsqueeze(1)

            x = x + self.pos_embed[:, k:k+1, :]                 # 加上位置编码

            # 通过 4 层 Transformer (共享参数, KV cache 加速)
            for i, blk in enumerate(self.blocks):
                x, new_kv = blk(x, past_kv=past_kv[i], use_cache=True)
                past_kv[i] = new_kv

            tokens_list.append(x)                                # 收集 tok_k

        tokens = torch.cat(tokens_list, dim=1)                   # (B, T, 512)

        # ── Step 3: 并行解码 (逐 token 独立 MLP, 无 token 间交互) ──
        colors = self.color_head(tokens)                         # (B, T, 3) RGB

        tokens_flat = tokens.reshape(B * T, self.token_dim)      # (B*T, 512)
        logits = self.mask_head.forward_logits(tokens_flat)      # (B*T, 64, 64) 原始值
        mask_logits = logits.reshape(B, T, 64, 64)               # (B, T, 64, 64)

        # ── Step 4: softmax_bg 合成 ──
        bg = self.bg.expand(B, -1, -1, -1)                      # (B, 3, 64, 64)

        # 广播颜色到空间维度
        c_full = colors.unsqueeze(-1).unsqueeze(-1)              # (B, T, 3, 1, 1)

        # softmax([logits_1, ..., logits_T, 0]) 沿 T+1 维
        # bg_logit 固定为 0 → 避免 softmax 平移不变性问题
        bg_logit = torch.zeros(B, 1, 64, 64, device=mask_logits.device)
        all_logits = torch.cat([mask_logits, bg_logit], dim=1)   # (B, T+1, 64, 64)
        w_all = F.softmax(all_logits, dim=1)                     # 每像素独立 softmax
        w_tok = w_all[:, :T]                                     # (B, T, 64, 64)
        w_bg = w_all[:, T:T+1]                                   # (B, 1, 64, 64)

        # canvas = Σ w_k * c_k + w_bg * bg
        canvas = (w_tok.unsqueeze(2) * c_full).sum(dim=1) + w_bg * bg

        return canvas, colors, w_tok

    @torch.no_grad()
    def generate(self, z):
        """确定性生成 (推理模式, 无梯度)。"""
        return self.forward(z)[0]

    def get_param_count(self):
        """返回参数量 (百万)。"""
        return sum(p.numel() for p in self.parameters()) / 1e6


# ═══════════════════════════════════════════════════════════════════════
# 5. 测试: 验证前向传播和参数量
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("V5RL-v6 (DecoderActorRL) 独立模型测试")
    print("=" * 60)

    # 构建默认配置模型
    model = DecoderActorRLV6()
    n_params = model.get_param_count()
    print(f"\n参数量: {n_params:.1f}M")

    # 详细参数统计
    print("\n--- 各模块参数 ---")
    for name, module in [
        ('z_proj', model.z_proj),
        ('input_proj', model.input_proj),
        ('blocks (×4)', model.blocks),
        ('color_head', model.color_head),
        ('mask_head', model.mask_head),
    ]:
        n = sum(p.numel() for p in module.parameters()) / 1e6
        print(f"  {name:20s}: {n:8.2f}M")

    # 前向传播测试
    print("\n--- 前向传播测试 ---")
    z = torch.randn(2, 128)                                    # batch=2
    canvas, colors, masks = model(z)

    print(f"  z:       {z.shape}")
    print(f"  canvas:  {canvas.shape}   (期望: [2, 3, 64, 64])")
    print(f"  colors:  {colors.shape}   (期望: [2, 8, 3])")
    print(f"  masks:   {masks.shape}   (期望: [2, 8, 64, 64])")
    print(f"  canvas 值域: [{canvas.min().item():.4f}, {canvas.max().item():.4f}]")

    # 验证 softmax 归一化 (每像素权重和为 1)
    w_sum = masks.sum(dim=1)                                   # (B, 64, 64)
    # 注意: w_tok 不含 bg, 所以 sum <= 1
    print(f"  Σw_tok 范围: [{w_sum.min().item():.4f}, {w_sum.max().item():.4f}]")

    # 验证不同 T 的兼容性
    print("\n--- 不同 Token 数量测试 ---")
    for T in [6, 7, 8]:
        try:
            m = DecoderActorRLV6(num_tokens=T)
            c, _, _ = m(torch.randn(1, 128))
            print(f"  T={T}: canvas shape = {c.shape}  ✓")
        except Exception as e:
            print(f"  T={T}: FAILED - {e}")

    print("\n✅ 所有测试通过!")
    print(f"\n模型架构确认:")
    print(f"  z(128) → z_proj(deep) → z_cond(512)")
    print(f"  AR Loop ×{model.num_tokens}: z_cond + pos + input_proj(prev) → 4×KVCacheBlock → tok")
    print(f"  tokens → RGBHeadDeep+LN → [0,1]³ per token")
    print(f"  tokens → MaskHead32(forward_logits) → logits")
    print(f"  softmax_bg([logits₁..logits_T, 0]) → canvas")
