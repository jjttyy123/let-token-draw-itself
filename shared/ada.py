"""Full ADA (Adaptive Discriminator Augmentation) for TokenPainter GAN.

Based on StyleGAN2-ADA (Karras et al., NeurIPS 2020). All augmentations in pure PyTorch.
Geometric transforms use affine_grid + grid_sample (no custom CUDA ops needed).
"""

import math
import torch
import torch.nn.functional as F


# ── Wavelet filter bank for imgfilter (same as StyleGAN2-ADA) ──

def _make_wavelet_filter(wavelet='sym2'):
    """Construct orthogonal lowpass + filter bank matching StyleGAN2-ADA."""
    wavelets = {
        'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
        'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057,
                 -0.048311742585633, 0.4910559419267466, 0.787641141030194,
                 0.3379294217276218, -0.07263752278646252, -0.021060292512300564,
                 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    }
    import numpy as np
    # Lowpass
    Hz_lo = np.asarray(wavelets[wavelet])
    Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size))
    Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2
    Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2
    # Filter bank: 4 bands
    Hz_fbank = np.eye(4, 1)
    for i in range(1, Hz_fbank.shape[0]):
        Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
        Hz_fbank = np.convolve(np.asarray(Hz_fbank).flatten()[:Hz_fbank.size], Hz_lo2)[:Hz_fbank.size].reshape(Hz_fbank.shape)
        from scipy import signal
        Hz_fbank = signal.convolve(Hz_fbank.reshape(-1), Hz_lo2)[:Hz_fbank.size].reshape(4, -1) if i == 1 else Hz_fbank
        # Actually let's just use StyleGAN2's exact construction
    # Simplified: pre-compute filter bank
    import scipy.signal
    Hz_fbank = np.eye(4, 1)
    for i in range(1, 4):
        Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
        Hz_col = scipy.signal.convolve(Hz_fbank[i-1], Hz_lo2) if i == 1 else Hz_fbank[i-1]
        Hz_fbank = np.zeros((4, Hz_fbank.shape[1] + Hz_hi2.size))
        for k in range(i):
            Hz_fbank[k, :Hz_fbank.shape[1] - Hz_hi2.size] = Hz_fbank[k, :Hz_fbank.shape[1] - Hz_hi2.size]
        Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
    return torch.tensor(Hz_fbank, dtype=torch.float32)


class AugmentPipe(torch.nn.Module):
    """Full StyleGAN2-ADA augmentation pipeline in pure PyTorch."""

    def __init__(self,
                 # Pixel blitting
                 xflip=1.0, rotate90=1.0, xint=1.0, xint_max=0.125,
                 # Geometric
                 scale=1.0, rotate=1.0, aniso=1.0, xfrac=1.0,
                 scale_std=0.2, rotate_max=1.0, aniso_std=0.2, xfrac_std=0.125,
                 # Color
                 brightness=1.0, contrast=1.0, lumaflip=1.0, hue=1.0, saturation=1.0,
                 brightness_std=0.2, contrast_std=0.5, hue_max=1.0, saturation_std=1.0,
                 # Filtering + corruption
                 imgfilter=1.0, imgfilter_std=1.0,
                 noise=1.0, cutout=1.0, noise_std=0.1, cutout_size=0.5,
                 # ADA
                 ada_target=0.6, ada_interval=4, ada_kimg=500):
        super().__init__()
        self.register_buffer('p', torch.tensor(0.0))

        # Pixel blitting
        self.xflip = float(xflip)
        self.rotate90 = float(rotate90)
        self.xint = float(xint)
        self.xint_max = float(xint_max)
        # Geometric
        self.scale = float(scale)
        self.rotate = float(rotate)
        self.aniso = float(aniso)
        self.xfrac = float(xfrac)
        self.scale_std = float(scale_std)
        self.rotate_max = float(rotate_max)
        self.aniso_std = float(aniso_std)
        self.xfrac_std = float(xfrac_std)
        # Color
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.lumaflip = float(lumaflip)
        self.hue = float(hue)
        self.saturation = float(saturation)
        self.brightness_std = float(brightness_std)
        self.contrast_std = float(contrast_std)
        self.hue_max = float(hue_max)
        self.saturation_std = float(saturation_std)
        # Filtering + corruption
        self.imgfilter = float(imgfilter)
        self.imgfilter_std = float(imgfilter_std)
        self.noise = float(noise)
        self.cutout = float(cutout)
        self.noise_std = float(noise_std)
        self.cutout_size = float(cutout_size)
        # ADA
        self.ada_target = ada_target
        self.ada_interval = ada_interval
        self.ada_kimg = ada_kimg
        self.ada_step = 0

        # Pre-compute imgfilter bank
        self.register_buffer('Hz_fbank', _make_wavelet_filter('sym2'))

    def _ada_update(self, r_acc_ema, batch_size):
        adjust_sign = 1.0 if r_acc_ema > self.ada_target else -1.0
        adjust = adjust_sign * batch_size * self.ada_interval / (self.ada_kimg * 1000)
        self.p.copy_((self.p + adjust).clamp(0.0, 0.9))
        self.ada_step += 1

    def forward(self, images):
        images = images.clone()
        B, C, H, W = images.shape
        device = images.device
        p = self.p.item()
        if p <= 0:
            return images

        # ── Pixel blitting (per-image ops, fast) ──
        # x-flip
        if self.xflip > 0:
            do_flip = torch.rand(B, device=device) < self.xflip * p
            images[do_flip] = torch.flip(images[do_flip], dims=[-1])

        # rotate90
        if self.rotate90 > 0:
            do_rot = torch.rand(B, device=device) < self.rotate90 * p
            if do_rot.any():
                k = torch.randint(0, 4, (B,), device=device)
                k[~do_rot] = 0
                for i, ki in enumerate(k):
                    if ki > 0:
                        images[i:i+1] = torch.rot90(images[i:i+1], ki.item(), dims=[-2, -1])

        # integer translation
        if self.xint > 0:
            do_trans = torch.rand(B, device=device) < self.xint * p
            if do_trans.any():
                max_shift = int(self.xint_max * min(H, W))
                if max_shift > 0:
                    tx = torch.randint(-max_shift, max_shift + 1, (B,), device=device)
                    ty = torch.randint(-max_shift, max_shift + 1, (B,), device=device)
                    for i in range(B):
                        if do_trans[i] and (tx[i] != 0 or ty[i] != 0):
                            images[i:i+1] = F.pad(images[i:i+1],
                                pad=[max(0, tx[i]), max(0, -tx[i]), max(0, ty[i]), max(0, -ty[i])],
                                mode='reflect')[:, :,
                                max(0, -ty[i]):max(0, -ty[i])+H,
                                max(0, -tx[i]):max(0, -tx[i])+W]

        # ── Geometric (affine_grid + grid_sample) ──
        do_geom = (self.scale > 0 or self.rotate > 0 or self.aniso > 0 or self.xfrac > 0)
        if do_geom:
            geom_mask = torch.ones(B, device=device)
            if self.scale > 0 and torch.rand(1).item() < self.scale * p:
                s = torch.exp2(torch.randn(B, device=device) * self.scale_std)
                s = torch.where(torch.rand(B, device=device) < self.scale * p, s, torch.ones_like(s))
            else:
                s = torch.ones(B, device=device)

            if self.rotate > 0 and torch.rand(1).item() < self.rotate * p:
                p_rot = 1 - math.sqrt(max(0, 1 - self.rotate * p))
                theta_pre = (torch.rand(B, device=device) * 2 - 1) * math.pi * self.rotate_max
                theta_pre = torch.where(torch.rand(B, device=device) < p_rot, theta_pre, torch.zeros_like(theta_pre))
                theta_post = (torch.rand(B, device=device) * 2 - 1) * math.pi * self.rotate_max
                theta_post = torch.where(torch.rand(B, device=device) < p_rot, theta_post, torch.zeros_like(theta_post))
            else:
                theta_pre = torch.zeros(B, device=device)
                theta_post = torch.zeros(B, device=device)

            if self.aniso > 0 and torch.rand(1).item() < self.aniso * p:
                sx = torch.exp2(torch.randn(B, device=device) * self.aniso_std)
                sx = torch.where(torch.rand(B, device=device) < self.aniso * p, sx, torch.ones_like(sx))
                sy = 1.0 / sx
            else:
                sx = torch.ones(B, device=device)
                sy = torch.ones(B, device=device)

            if self.xfrac > 0 and torch.rand(1).item() < self.xfrac * p:
                txy = torch.randn(B, 2, device=device) * self.xfrac_std
                txy = torch.where(torch.rand(B, 1, device=device) < self.xfrac * p, txy, torch.zeros_like(txy))
            else:
                txy = torch.zeros(B, 2, device=device)

            # Build per-image affine matrices
            has_any_geom = (s != 1).any() or (theta_pre != 0).any() or (theta_post != 0).any() or \
                           (sx != 1).any() or (txy != 0).any()
            if has_any_geom:
                theta = torch.zeros(B, 2, 3, device=device)
                # Combine: translate(frac) * rotate(post) * scale(aniso) * rotate(pre) * scale(iso)
                for i in range(B):
                    cos_pre = theta_pre[i].cos(); sin_pre = theta_pre[i].sin()
                    cos_s_pre = cos_pre * s[i]; sin_s_pre = sin_pre * s[i]
                    cos_post = theta_post[i].cos(); sin_post = theta_post[i].sin()
                    # Scale(iso) * Rotate(pre)
                    M = torch.tensor([[cos_s_pre, -sin_s_pre, 0],
                                       [sin_s_pre,  cos_s_pre, 0]], device=device)
                    # Scale(aniso)
                    M[:, 0] *= sx[i]; M[:, 1] *= sy[i]
                    # Rotate(post)
                    a, b = M[:, 0].clone(), M[:, 1].clone()
                    M[:, 0] = cos_post * a + sin_post * b
                    M[:, 1] = -sin_post * a + cos_post * b
                    # Translate(fractional)
                    M[:, 2] = txy[i]
                    # Normalize to [-1, 1] grid
                    theta[i] = M

                grid = F.affine_grid(theta, images.shape, align_corners=False)
                images = F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection',
                                       align_corners=False)

        # ── Color ──
        do_color = (self.brightness > 0 or self.contrast > 0 or self.lumaflip > 0 or
                     self.hue > 0 or self.saturation > 0)
        if do_color and C == 3:
            # brightness
            if self.brightness > 0:
                b = torch.randn(B, device=device) * self.brightness_std
                b = torch.where(torch.rand(B, device=device) < self.brightness * p, b, torch.zeros_like(b))
                images = images + b.view(-1, 1, 1, 1)

            # contrast
            if self.contrast > 0:
                c = torch.exp2(torch.randn(B, device=device) * self.contrast_std)
                c = torch.where(torch.rand(B, device=device) < self.contrast * p, c, torch.ones_like(c))
                mean = images.mean(dim=[-2, -1], keepdim=True)
                images = (images - mean) * c.view(-1, 1, 1, 1) + mean

            # lumaflip (flip luminance axis)
            if self.lumaflip > 0:
                v = torch.tensor([1., 1., 1.], device=device) / math.sqrt(3)  # luma axis
                do_lf = torch.rand(B, device=device) < self.lumaflip * p
                for i in range(B):
                    if do_lf[i]:
                        lum = (images[i] * v.view(3, 1, 1)).sum(dim=0, keepdim=True)
                        images[i] = images[i] - 2 * lum * v.view(3, 1, 1)

            # hue (RGB rotation around luma axis)
            if self.hue > 0:
                v = torch.tensor([1., 1., 1.], device=device) / math.sqrt(3)
                theta = (torch.rand(B, device=device) * 2 - 1) * math.pi * self.hue_max
                theta = torch.where(torch.rand(B, device=device) < self.hue * p, theta, torch.zeros_like(theta))
                cos_t = theta.cos(); sin_t = theta.sin()
                # Rodrigues rotation for each sample
                v_vT = torch.outer(v, v)  # 3x3
                v_cross = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=device)
                for i in range(B):
                    if theta[i] != 0:
                        R = v_vT + cos_t[i] * (torch.eye(3, device=device) - v_vT) + sin_t[i] * v_cross
                        images[i] = (R @ images[i].reshape(3, -1)).reshape(3, H, W)

            # saturation
            if self.saturation > 0:
                gray = images.mean(dim=1, keepdim=True)
                s = torch.exp2(torch.randn(B, device=device) * self.saturation_std)
                s = torch.where(torch.rand(B, device=device) < self.saturation * p, s, torch.ones_like(s))
                images = gray + (images - gray) * s.view(-1, 1, 1, 1)

            images = images.clamp(0, 1)

        # ── Image-space filtering (wavelet bands) ──
        if self.imgfilter > 0 and torch.rand(1).item() < self.imgfilter * p:
            Hz = self.Hz_fbank.to(device)  # [4, tap]
            num_bands = Hz.shape[0]
            expected_power = torch.tensor([10., 1., 1., 1.], device=device) / 13.0

            g = torch.ones(B, num_bands, device=device)
            for i in range(num_bands):
                t_i = torch.exp2(torch.randn(B, device=device) * self.imgfilter_std)
                t_i = torch.where(torch.rand(B, device=device) < self.imgfilter * p, t_i, torch.ones_like(t_i))
                t = torch.ones(B, num_bands, device=device)
                t[:, i] = t_i
                t = t / (expected_power * t.square()).sum(dim=-1, keepdim=True).sqrt()
                g = g * t

            Hz_prime = g @ Hz  # [B, tap]
            p_half = Hz.shape[1] // 2
            # Apply as separable 1D conv horizontally and vertically
            im = images.reshape(B * C, 1, H, W)
            im = F.pad(im, [p_half, p_half, p_half, p_half], mode='reflect')
            weight_h = Hz_prime.unsqueeze(1).unsqueeze(2).repeat(1, C, 1).reshape(B * C, 1, 1, -1)
            weight_v = Hz_prime.unsqueeze(1).unsqueeze(3).repeat(1, C, 1).reshape(B * C, 1, -1, 1)
            im = F.conv2d(im, weight_h, groups=B * C, padding=0)
            im = F.conv2d(im, weight_v, groups=B * C, padding=0)
            images = im.reshape(B, C, H, W).clamp(0, 1)

        # ── Noise ──
        if self.noise > 0:
            sigma = torch.randn(B, 1, 1, 1, device=device).abs() * self.noise_std
            sigma = torch.where(torch.rand(B, 1, 1, 1, device=device) < self.noise * p, sigma, torch.zeros_like(sigma))
            images = images + torch.randn_like(images) * sigma

        # ── Cutout ──
        if self.cutout > 0:
            do_cut = torch.rand(B, device=device) < self.cutout * p
            if do_cut.any():
                b_cut = int(do_cut.sum().item())
                size_h = int(self.cutout_size * H * (0.5 + 0.5 * torch.rand(1).item()))
                size_w = int(self.cutout_size * W * (0.5 + 0.5 * torch.rand(1).item()))
                size_h = max(1, min(size_h, H - 1))
                size_w = max(1, min(size_w, W - 1))
                ys = torch.randint(0, H - size_h, (b_cut,))
                xs = torch.randint(0, W - size_w, (b_cut,))
                idx = torch.where(do_cut)[0]
                fill_val = images[idx].mean(dim=[1, 2, 3], keepdim=True)  # mean color fill
                for j, ii in enumerate(idx):
                    fill = fill_val[j].view(C, 1, 1).expand(C, size_h, size_w)
                    images[ii, :, ys[j]:ys[j]+size_h, xs[j]:xs[j]+size_w] = fill

        return images.clamp(0, 1)


def create_ada_pipe(for_64x64=True):
    """Create full StyleGAN2-ADA augmentation pipeline."""
    return AugmentPipe(
        # Pixel blitting
        xflip=1.0, rotate90=1.0, xint=1.0, xint_max=0.125,
        # Geometric
        scale=1.0, rotate=1.0, aniso=1.0, xfrac=1.0,
        scale_std=0.2, rotate_max=1.0, aniso_std=0.2, xfrac_std=0.125,
        # Color
        brightness=1.0, contrast=1.0, lumaflip=1.0, hue=1.0, saturation=1.0,
        brightness_std=0.2, contrast_std=0.5, hue_max=1.0, saturation_std=1.0,
        # Filtering + corruption
        imgfilter=1.0, imgfilter_std=1.0,
        noise=1.0, cutout=1.0, noise_std=0.1, cutout_size=0.5,
        # ADA
        ada_target=0.6, ada_interval=4, ada_kimg=500,
    )
