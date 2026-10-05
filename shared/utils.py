"""Shared utilities for TokenPainter MSE training scripts."""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def to_np(t):
    """Tensor (C,H,W) or (H,W,C) → uint8 (H,W,C)."""
    t = t.detach().clamp(0, 1)
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return t.cpu().mul(255).byte().numpy()


def make_grid(imgs, ncols, gap=2):
    """List of (H,W,3) uint8 → PIL Image grid."""
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    nrows = (len(imgs) + ncols - 1) // ncols
    H = nrows * h + (nrows - 1) * gap
    W = ncols * w + (ncols - 1) * gap
    canvas = np.full((H, W, 3), 128, dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = i // ncols, i % ncols
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        canvas[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = img
    return Image.fromarray(canvas)


def ssim_psnr(pred, gt):
    """Compute SSIM and PSNR for monitoring."""
    mse = F.mse_loss(pred, gt)
    psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(100.0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = pred.mean(dim=(-2, -1))
    mu_y = gt.mean(dim=(-2, -1))
    var_x = ((pred - mu_x.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))
    var_y = ((gt - mu_y.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))
    cov = ((pred - mu_x.unsqueeze(-1).unsqueeze(-1)) *
           (gt - mu_y.unsqueeze(-1).unsqueeze(-1))).mean(dim=(-2, -1))
    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return ssim_val.mean().item(), psnr.item()


def plot_metrics(metrics_history, save_path):
    """Training curves: loss + SSIM + PSNR. PDF vector output."""
    epochs = [m['epoch'] for m in metrics_history]
    losses = [m['loss'] for m in metrics_history]
    ssims = [m['ssim'] for m in metrics_history]
    psnrs = [m['psnr'] for m in metrics_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, losses, 'b-', linewidth=1.0)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, ssims, 'g-', linewidth=1.0)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, psnrs, 'r-', linewidth=1.0)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('SSIM', color='g')
    ax2_twin.set_ylabel('PSNR (dB)', color='r')
    ax2.set_title('Reconstruction Quality'); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
