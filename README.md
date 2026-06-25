# Let Token Draw Itself

Reference-free pixel art generation using reinforcement learning on a consumer GPU.
The Actor never sees any real image during training — it receives only scalar reward signals
from a co-trained SAGAN reward model.

Each token independently decides its own color and spatial mask, then all tokens compose
via softmax-bg into the final image — **let tokens draw themselves.**

## Overview

![Architecture](https://via.placeholder.com/800x300?text=V5RL-R7+Architecture)

A 128-dim latent vector $z$ is transformed by a 4-layer Transformer decoder into a sequence of
tokens. Each token is decoded into an RGB color and a spatial mask. All tokens are composited
via softmax-bg (a convex combination with zero new parameters) to produce a 64×64 image.

**Key idea:** the Actor learns to generate pixel art purely from a reward model's scalar feedback,
without ever seeing a real image.

## Architecture (V5RL-R7)

```
z(128) → z_proj(2-layer MLP) → z_cond(512)
  ↓
For each step k = 0..T-1:
  x = z_cond + pos_embed[k] + input_proj(tok_{k-1})
  x → Transformer×4 + KV-cache → tok_k
  tok_k → RGBHeadDeep+LN → c_k ∈ [0,1]³
  tok_k → MaskHead32 → logits_k ∈ R^{64×64}
  ↓
softmax_bg([logits₁,...,logits_T, 0]) → canvas ∈ [0,1]^{3×64×64}
```

- Transformer: 4 layers, 4 heads, d_k=128, FFN 512→4096→512, GELU
- RGBHead: 3-layer residual MLP with LayerNorm
- MaskHead: FC(512→128×32×32) → ConvTranspose → Conv2d
- ~90M parameters

## Training

### Phase 1: MSE Pre-training
Fixed z-image pairs on TinyHero (2K images), optimizing reconstruction with MSE loss.
Achieves MSE = 6.1×10⁻⁵ at R7 configuration.

### Phase 2: Adversarial RL
SAGAN discriminator as co-trained reward model with Hinge Loss + R1 penalty.
Actor trained with GROUP_POS: group-normalized advantages with non-negative
gradients ("reward-only"), using continuous gradient directly through the SAGAN score.

```
GROUP_POS:
  adv = (score - μ_group) / σ_group
  adv = max(0, adv)              # reward only, no penalty
  loss = -(adv * score).mean()   # maximize scores via gradient descent
```

## Key Results

- **Token interpretability:** each token (color + mask) has clear semantic roles
- **T extrapolation:** model trained at T=4 generalizes to T=16
- **Multi-view:** view-conditioned reward model guides unconditional Actor

## File Structure

```
let-token-draw-itself/
├── README.md
├── requirements.txt
├── models/
│   ├── v5rl_r7.py               # V5RL-R7 Actor (standalone)
│   └── sagan/                    # SAGAN reward model
│       ├── __init__.py
│       ├── generator.py
│       ├── discriminator.py
│       └── layers.py
├── training/
│   ├── train_rl.py               # Main RL training (DDPG/GROUP)
│   ├── train_mse.py              # MSE pre-training
│   └── visualize.py              # Token visualization
└── results/
    └── samples/                   # Generated examples
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# MSE pre-training
python training/train_mse.py

# RL training with SAGAN reward model
python training/train_rl.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on RTX 3060 Laptop 6GB)
- See `requirements.txt` for full list

## Citation

```bibtex
@misc{tang2026lettoken,
  title={Let Token Draw Itself: Reference-Free Pixel Art Generation via Autoregressive RL},
  author={Tang, Yujing},
  year={2026}
}
```

## License

MIT
