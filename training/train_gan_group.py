"""TokenPainter GAN training with GroupPos loss.

Key idea: Global advantage normalization: z-score across all ga_g × g_batch samples.
K candidates per group compete internally — only above-group-mean get gradient.
D trains on separate random z (diverse negatives), independent from G's batch.

Usage:
  python training/train_gan_group.py --data metfaces --model tp64 \
      --num_tokens 4 --df_dim 96 --g_batch 56 --d_batch 112 --ga_g 2 --ga_d 1 \
      --kimg 1000 --lr 2e-4 --lr_ratio 1.0 --g_lr 4e-4 --beta1 0.0 --beta2 0.9 \
      --ada --bf16 --r1_gamma 100 --r2_gamma 100 --r1r2_interval 4 \
      --ema --ema_kimg 10.0 --ckpt_interval 100 --seed 42 --run_name my_run
"""
import os, sys, time, argparse, json, math, psutil
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent  # repo root containing TokenPainter/, sagan/, shared/
sys.path.insert(0, str(_ROOT_DIR))
from TokenPainter.model import TokenPainter64, TokenPainter64Heavy, TokenPainter32
from sagan.discriminator import UnconditionalDiscriminator
from shared.utils import to_np, make_grid
from shared.ada import AugmentPipe


# ── Helpers ──

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


# Dataset registry: short name -> folder of PNG/JPG images.
# Anything not in this dict is treated as a path, so `--data /my/images` also works.
DATA_PATHS = {
    'tinyhero': './data/tinyhero',
    'metfaces': './data/metfaces_64',
    'celeba':   './data/celeba_64',
    'celeba32': './data/celeba_32',
    'cifar10':  'cifar10',
}


def load_images(data_dir, img_size=64):
    data_dir = Path(os.path.expanduser(data_dir))
    files = sorted(data_dir.rglob('*.png')) or sorted(data_dir.rglob('*.jpg'))
    if not files: files = sorted(data_dir.glob('*.png')) or sorted(data_dir.glob('*.jpg'))
    imgs = [np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0 for f in files]
    return torch.stack([torch.from_numpy(a.transpose(2, 0, 1))[:, :img_size, :img_size] for a in imgs]).clamp(0, 1)


def cosine_decay(cur_nimg, base_value, final_value, total_nimg, warmup_nimg=0):
    if cur_nimg < warmup_nimg:
        return base_value * cur_nimg / max(warmup_nimg, 1)
    if cur_nimg > total_nimg: return final_value
    progress = (cur_nimg - warmup_nimg) / float(total_nimg - warmup_nimg)
    return final_value + (base_value - final_value) * 0.5 * (1 + math.cos(math.pi * progress))


def dnnlib_format_time(seconds):
    d, h = int(seconds // 86400), int((seconds % 86400) // 3600)
    m, s = int((seconds % 3600) // 60), int(seconds % 60)
    if d > 0: return f'{d}d {h:02d}:{m:02d}:{s:02d}'
    return f'{h:02d}:{m:02d}:{s:02d}'


# Old ada_augment removed — using AugmentPipe from shared.ada instead


def make_token_viz(canvas_batch, colors_batch, masks_batch, T):
    """3-row token decomposition: colors / masks / accumulation."""
    B = min(2, canvas_batch.size(0)); H, W = canvas_batch.shape[2], canvas_batch.shape[3]
    all_imgs = []
    for b in range(B):
        c = canvas_batch[b].clamp(0, 1).cpu(); vc = colors_batch[b].clamp(0, 1).cpu(); vm = masks_batch[b].clamp(0, 1).cpu()
        canvas_np = to_np(c)
        # Row 1: canvas + color swatches
        row1 = [canvas_np] + [np.full((H, W, 3), (vc[k, 0].item() * 255, vc[k, 1].item() * 255, vc[k, 2].item() * 255), dtype=np.uint8) for k in range(T)]
        # Row 2: canvas + mask overlays
        row2 = [canvas_np] + [(vm[k].numpy()[:, :, None].repeat(3, axis=2) * 255).astype(np.uint8) for k in range(T)]
        # Row 3: canvas + accumulation progression
        acc = torch.zeros(3, H, W); row3 = [canvas_np]
        for k in range(T):
            acc += vm[k:k + 1] * vc[k].unsqueeze(-1).unsqueeze(-1); row3.append(to_np(acc))
        all_imgs.extend(row1 + row2 + row3)
    return make_grid(all_imgs, T + 1)


class InfiniteSampler:
    def __init__(self, N, seed=0, window_size=0.5):
        self.N = N; self.rng = np.random.RandomState(seed)
        self.order = np.arange(N); self.rng.shuffle(self.order)
        self.window = max(2, int(N * window_size)); self.idx = 0

    def next(self, B):
        indices = []
        for _ in range(B):
            i = self.idx % self.N; indices.append(self.order[i])
            if self.window >= 2:
                j = (i - self.rng.randint(self.window)) % self.N
                self.order[i], self.order[j] = self.order[j], self.order[i]
            self.idx += 1
        return indices


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='metfaces'); parser.add_argument('--run_name', default='GAN_group')
    parser.add_argument('--model', default='tp64', choices=['tp64', 'tp64heavy', 'tp32'])
    parser.add_argument('--num_tokens', type=int, default=8)
    parser.add_argument('--token_dim', type=int, default=512)
    parser.add_argument('--df_dim', type=int, default=64, help='D channel width')
    parser.add_argument('--g_batch', type=int, default=64, help='G images per micro-batch (g_batch/K groups)')
    parser.add_argument('--d_batch', type=int, default=156, help='D images per micro-batch (real+fake each)')
    parser.add_argument('--ga_g', type=int, default=4, help='G gradient accumulation steps')
    parser.add_argument('--ga_d', type=int, default=1, help='D gradient accumulation steps')
    parser.add_argument('--d_freq', type=int, default=1, help='Update D every N steps (1=every step)')
    parser.add_argument('--kimg', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=2e-4, help='Base LR (for D, unless G LR set separately)')
    parser.add_argument('--lr_ratio', type=float, default=1.0,
                        help='Final-to-initial LR ratio (1.0=constant, 0.25=cosine decay to 25%%)')
    parser.add_argument('--g_lr', type=float, default=0.0,
                        help='G learning rate (0=use --lr for both)')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Adam beta1 for both G and D (0=off, standard for GAN)')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Adam beta2 for both G and D (0.99=StyleGAN2, 0.9=more adaptive)')
    parser.add_argument('--g_dropout', type=float, default=0.0,
                        help='Dropout in G transformer blocks (0=off)')
    parser.add_argument('--ada', action='store_true'); parser.add_argument('--ada_target', type=float, default=0.4)
    parser.add_argument('--ada_speed', type=int, default=500)
    parser.add_argument('--ada_p_min', type=float, default=0.0, help='Base augmentation probability (floor for ADA)')
    parser.add_argument('--d_gap_lower', type=float, default=0.5,
                        help='D trains normally below this d_gap')
    parser.add_argument('--d_gap_upper', type=float, default=0.0,
                        help='D LR reduction ceiling (0=off, uses constant LR per paper)')
    parser.add_argument('--d_loss', default='softplus', choices=['hinge', 'softplus'],
                        help='D loss type')
    parser.add_argument('--g_loss', default='group_pos', choices=['group_pos', 'softplus'],
                        help='G loss: group_pos=positive-advantage only (global z-score), softplus=non-saturating -F.softplus(-scores).mean()')
    parser.add_argument('--adv_temp', type=float, default=2.0,
                        help='Temperature for group_pos soft-threshold (lower=flatter weights)')
    parser.add_argument('--bf16', action='store_true'); parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--snap', type=int, default=10)
    parser.add_argument('--ckpt_interval', type=int, default=500)
    parser.add_argument('--tick_interval', type=int, default=5)
    parser.add_argument('--r1_gamma', type=float, default=0.0, help='R1 gradient penalty weight')
    parser.add_argument('--r2_gamma', type=float, default=0.0, help='R2 gradient penalty weight (fake images)')
    parser.add_argument('--r1r2_interval', type=int, default=16, help='Lazy R1/R2: compute every N steps')
    parser.add_argument('--use_sn', action='store_true', help='Spectral Normalization in D')
    parser.add_argument('--ema', action='store_true', help='EMA of G weights for inference (StyleGAN2-style)')
    parser.add_argument('--ema_kimg', type=float, default=10.0, help='EMA half-life in kimg (default 10, like StyleGAN2)')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='Number of classes for conditional generation (0=unconditional)')
    parser.add_argument('--label_embed_dim', type=int, default=128,
                        help='Label embedding dimension in G')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_tb', action='store_true')
    args = parser.parse_args()

    # Fix random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_BF16 = args.bf16 and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    print(f'[Setup] bf16={USE_BF16} D_loss={args.d_loss} G_loss={args.g_loss} device={device}')

    # ADA augmentation pipeline (full StyleGAN2-ADA)
    augment_pipe = AugmentPipe().to(device) if args.ada else None
    if augment_pipe:
        print(f'[ADA] Full StyleGAN2-ADA pipeline, target={args.ada_target} speed={args.ada_speed}')

    # Output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = _ROOT_DIR / 'runs' / f'{args.run_name}_{ts}'
    img_dir = run_dir / 'images'; run_dir.mkdir(parents=True, exist_ok=True); img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / 'fakes').mkdir(exist_ok=True); (img_dir / 'tokens').mkdir(exist_ok=True)
    (img_dir / 'best_fake_g').mkdir(exist_ok=True); (img_dir / 'best_fake_d').mkdir(exist_ok=True)

    # Logging (before data loading)
    _console = open(run_dir / 'console.log', 'w')
    sys.stdout = Tee(sys.stdout, _console)
    sys.stderr = Tee(sys.stderr, _console)
    log_f = open(run_dir / 'train.log', 'w')
    tb_writer = None
    if not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(run_dir)
        except ImportError: pass

    # Data
    img_size = 32 if args.model == 'tp32' else 64
    train_labels = None
    if args.data == 'cifar10':
        import torchvision
        trainset = torchvision.datasets.CIFAR10(root=os.path.expanduser('~/data'), train=True, download=False,
                                                transform=tvf.Compose([tvf.ToTensor()]))
        images = torch.stack([x[0] for x in trainset])
        train_labels = torch.tensor([x[1] for x in trainset], dtype=torch.long, device=device)
        print(f'[Data] CIFAR-10: {len(images)} images, {args.num_classes} classes')
    else:
        images = load_images(DATA_PATHS.get(args.data, args.data), img_size)
    N = len(images); images = images.to(device)
    total_kimg = args.kimg; total_nimg = total_kimg * 1000

    g_batch = args.g_batch; d_batch = args.d_batch
    ga_g = args.ga_g; ga_d = args.ga_d
    g_nimg_per_step = ga_g * g_batch  # kimg = total G forward images per optimizer step

    print(f'[Data] {args.data}: {N} images, {img_size}x{img_size}')
    print(f'[Training] G_B={g_batch} D_B={d_batch} ga_g={ga_g} ga_d={ga_d} kimg={total_kimg} nimg={total_nimg}')

    # Models
    g_dropout = args.g_dropout
    nc = args.num_classes; led = args.label_embed_dim
    if args.model == 'tp32':
        G = TokenPainter32(num_tokens=args.num_tokens, token_dim=args.token_dim,
                           dropout=g_dropout, num_classes=nc, label_embed_dim=led).to(device)
    elif args.model == 'tp64heavy':
        G = TokenPainter64Heavy(num_tokens=args.num_tokens,
                                token_dim=args.token_dim, num_classes=nc, label_embed_dim=led).to(device)
    else:
        G = TokenPainter64(num_tokens=args.num_tokens,
                           token_dim=args.token_dim,
                           dropout=g_dropout, num_classes=nc, label_embed_dim=led).to(device)
    from sagan.discriminator import ConditionalDiscriminator
    D = ConditionalDiscriminator(df_dim=args.df_dim, use_sn=args.use_sn, num_classes=nc).to(device)
    n_g = sum(p.numel() for p in G.parameters()) / 1e6
    n_d = sum(p.numel() for p in D.parameters()) / 1e6
    print(f'[Model] G={n_g:.1f}M D={n_d:.1f}M T={args.num_tokens}')
    print(f'[G Structure]\n{G}')
    print(f'[D Structure]\n{D}')

    g_lr = args.g_lr if args.g_lr > 0 else args.lr
    opt_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(args.beta1, args.beta2), fused=True)
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), fused=True)

    # EMA of G weights for inference (StyleGAN2-style, half-life in kimg)
    G_ema = None
    ema_beta = None
    if args.ema:
        G_ema = {k: v.detach().clone() for k, v in G.state_dict().items()}
        g_images_per_step = args.ga_g * args.g_batch
        ema_half_life_steps = args.ema_kimg * 1000 / g_images_per_step
        ema_beta = 0.5 ** (1.0 / ema_half_life_steps)
        print(f'[EMA] half-life={args.ema_kimg}kimg ({ema_half_life_steps:.0f} steps), beta={ema_beta:.6f}')

    sampler = InfiniteSampler(N, seed=args.seed)
    if nc > 0:
        per_class = max(1, 100 // nc)  # keep total ~100 images
        n_viz = nc * per_class
        fixed_z_viz = torch.randn(n_viz, 128, device=device)
        fixed_y_viz = torch.arange(nc, device=device).repeat_interleave(per_class)
    else:
        fixed_z_viz = torch.randn(64, 128, device=device)
        fixed_y_viz = None

    # ADA state
    ada_p = args.ada_p_min; ada_rt = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt['G'], strict=False)
        D.load_state_dict(ckpt['D'], strict=False)
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        g_nimg_resume = ckpt['nimg']
        d_nimg_resume = ckpt.get('d_nimg', g_nimg_resume)
        d_step_resume = ckpt.get('d_step', 0)
        ada_p = ckpt.get('ada_p', 0.0); ada_rt = ckpt.get('ada_rt', 0.0)
        # Restore EMA from checkpoint
        if G_ema is not None and 'G_ema' in ckpt:
            for k, v in ckpt['G_ema'].items():
                G_ema[k].copy_(v)
        print(f'[Resume] Restored from {g_nimg_resume/1000:.1f}k, d_nimg={d_nimg_resume/1000:.1f}k, ada_p={ada_p:.3f} ada_rt={ada_rt:.3f}')
        del ckpt
    else:
        g_nimg_resume = 0; d_nimg_resume = 0; d_step_resume = 0

    cfg = {}
    for k, v in vars(args).items():
        cfg[k] = str(v) if not isinstance(v, (int, float, bool)) else v
    json.dump(cfg, open(run_dir / 'config.json', 'w'), indent=2)

    t_start = time.time(); g_nimg = g_nimg_resume; d_nimg = d_nimg_resume; step = 0; d_step = d_step_resume
    tick_start_time = t_start; tick_start_nimg = 0
    d_loss_main = 0.0; d_gap = 0.0; d_lr_scale = 1.0; r_acc = 0.0; f_acc = 0.0
    g_mean = 0.0; g_worst = 0.0; g_best = 0.0; r1_val = 0.0; r2_val = 0.0
    d_real_mean = 0.0; d_fake_mean = 0.0; d_fake_max = 0.0
    adv_pos_ratio = 0.5; adv_mean_val = 0.0; adv_std_val = 0.0

    print(f'[Training] Starting {total_kimg} kimg...')

    while True:
        if g_nimg >= total_nimg: break
        G.train(); D.train()

        cur_lr_g = cosine_decay(g_nimg, g_lr, g_lr * args.lr_ratio, total_nimg)
        cur_lr_d = cosine_decay(d_nimg, args.lr, args.lr * args.lr_ratio, total_nimg)
        for pg in opt_G.param_groups: pg['lr'] = cur_lr_g
        for pg in opt_D.param_groups: pg['lr'] = cur_lr_d

        # ── G phase ──
        if args.g_loss == 'group_pos':
            # Two-pass with SAME z: Phase 1 (no_grad) collects stats,
            # Phase 2 (with_grad) uses same z, backprop one micro-batch at a time.
            # This keeps only 1 computation graph alive → memory efficient.

            # Phase 1: collect scores and save z (no grad)
            G.eval(); G.requires_grad_(False); D.requires_grad_(False)
            z_saved = []
            all_scores_list = []
            g_best_fakes = []
            with torch.no_grad():
                for gi in range(ga_g):
                    z_g = torch.randn(g_batch, 128, device=device)
                    y_g = torch.randint(0, nc, (g_batch,), device=device) if nc > 0 else None
                    z_saved.append((z_g, y_g))
                    with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                        fake_g, _, _ = G(z_g, y_g)
                        s = D(fake_g, y_g).flatten()
                    all_scores_list.append(s)
                    s_flat = s.flatten()
                    for i in range(g_batch):
                        g_best_fakes.append((s_flat[i].item(), fake_g[i].detach().clone().cpu()))

            # Global stats from detached scores (kept on GPU, no CPU sync)
            scores_cat = torch.cat(all_scores_list)  # [ga_g * g_batch]
            global_mean = scores_cat.mean().detach()
            global_std = scores_cat.std().detach() + 1e-8

            # Phase 2: forward with grad, SAME z, soft threshold, per-micro-batch backward
            G.train(); G.requires_grad_(True); D.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)
            T = args.adv_temp  # temperature for softplus soft-threshold
            g_gan_total = torch.tensor(0.0, device=device)
            adv_pos_ratios, adv_mean_vals, adv_std_vals = [], [], []
            for gi in range(ga_g):
                z_g, y_g = z_saved[gi]
                with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                    fake_g, _, _ = G(z_g, y_g)
                    scores = D(fake_g, y_g).flatten()
                advantages = (scores - global_mean) / global_std
                adv_flat = advantages.detach()
                soft_adv = F.softplus(adv_flat * T) / T
                g_gan_val = -(soft_adv * scores).mean()
                (g_gan_val / ga_g).backward()
                g_gan_total = g_gan_total + g_gan_val.detach()
                adv_pos_ratios.append((advantages > 0).float().mean().item())
                adv_mean_vals.append(advantages.mean().item())
                adv_std_vals.append(advantages.std().item())

            del z_saved
        else:
            # ── softplus: non-saturating G loss, single-pass ──
            G.train(); G.requires_grad_(True); D.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)
            g_gan_total = torch.tensor(0.0, device=device)
            g_best_fakes = []
            all_scores_list = []
            adv_pos_ratios, adv_mean_vals, adv_std_vals = [], [], []
            for gi in range(ga_g):
                z_g = torch.randn(g_batch, 128, device=device)
                y_g = torch.randint(0, nc, (g_batch,), device=device) if nc > 0 else None
                with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                    fake_g, _, _ = G(z_g, y_g)
                    scores = D(fake_g, y_g).flatten()  # [g_batch]
                g_gan_val = F.softplus(-scores).mean()
                (g_gan_val / ga_g).backward()
                g_gan_total = g_gan_total + g_gan_val.detach()
                s_d = scores.detach()
                all_scores_list.append(s_d)
                s_flat = s_d.flatten()
                for i in range(g_batch):
                    g_best_fakes.append((s_flat[i].item(), fake_g[i].detach().clone().cpu()))
                adv_pos_ratios.append(0.0)
                adv_mean_vals.append(0.0)
                adv_std_vals.append(0.0)

        for p in G.parameters():
            if p.grad is not None: torch.nan_to_num(p.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=p.grad)
        torch.nn.utils.clip_grad_norm_(G.parameters(), 0.5)
        opt_G.step()

        # EMA update
        if G_ema is not None:
            with torch.no_grad():
                for k, v in G.state_dict().items():
                    G_ema[k].mul_(ema_beta).add_(v.detach(), alpha=1 - ema_beta)

        g_gan_main = (g_gan_total / ga_g).item()
        adv_pos_ratio = sum(adv_pos_ratios) / len(adv_pos_ratios)
        adv_mean_val = sum(adv_mean_vals) / len(adv_mean_vals)
        adv_std_val = sum(adv_std_vals) / len(adv_std_vals)
        all_scores = torch.cat(all_scores_list)
        g_mean = all_scores.mean().item()
        g_worst = all_scores.min().item()
        del all_scores_list, all_scores
        g_best_fakes.sort(key=lambda x: x[0], reverse=True)
        g_best = g_best_fakes[0][0] if g_best_fakes else 0.0
        torch.cuda.empty_cache()

        # ── D phase: hinge with independent random z (D1G1, every step) ──
        d_best_fakes = []  # (score, image_cpu) for best_fake_d snapshot
        # Determine if D should run this step
        d_ran = step % args.d_freq == 0
        if d_ran:
            G.requires_grad_(False); D.requires_grad_(True)
            opt_D.zero_grad(set_to_none=True)
            d_loss_total = torch.tensor(0.0, device=device)
            d_real_last = None; d_fake_last = None
            r_signs = [] if args.ada else None

            for di in range(ga_d):
                ridx = sampler.next(d_batch)
                real = images[ridx]
                real_labels = train_labels[ridx] if train_labels is not None else None
                with torch.no_grad():
                    z_d = torch.randn(d_batch, 128, device=device)
                    y_fake_d = torch.randint(0, nc, (d_batch,), device=device) if nc > 0 else None
                    with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                        fake_d, _, _ = G(z_d, y_fake_d)

                if augment_pipe is not None:
                    augment_pipe.p.fill_(ada_p)
                    real_aug = augment_pipe(real)
                    augment_pipe.p.fill_(ada_p)
                    fake_aug = augment_pipe(fake_d)
                else:
                    real_aug = real; fake_aug = fake_d

                with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                    d_real = D(real_aug, real_labels)
                    d_fake = D(fake_aug, y_fake_d)
                if args.d_loss == 'hinge':
                    d_loss_val = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                else:
                    d_loss_val = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

                (d_loss_val / ga_d).backward()
                d_loss_total = d_loss_total + d_loss_val.detach()
                d_real_last = d_real; d_fake_last = d_fake
                # Use augmented D scores for ranking; save raw fake images
                d_sc = d_fake.detach().flatten()
                for i in range(d_batch):
                    d_best_fakes.append((d_sc[i].item(), fake_d[i].detach().clone().cpu()))
                if args.ada:
                    r_signs.append(d_real.detach().sign().mean())

            d_real_mean = d_real_last.mean().item(); d_fake_mean = d_fake_last.mean().item()
            d_fake_max = d_fake_last.max().item()
            d_gap = d_real_mean - d_fake_mean

            # Dual-threshold d_gap control: linearly reduce D LR from lower to upper.
            # d_gap < lower: D trains normally (d_lr_scale=1.0)
            # lower ≤ d_gap ≤ upper: d_lr_scale linearly 1.0 → 0.01
            # d_gap > upper: D LR clamped at 0.01 (almost frozen)
            d_lr_scale = 1.0
            if args.d_gap_upper > 0 and d_gap > args.d_gap_lower:
                t = min((d_gap - args.d_gap_lower) / (args.d_gap_upper - args.d_gap_lower), 1.0)
                d_lr_scale = 1.0 - 0.99 * t
                for pg in opt_D.param_groups:
                    pg['lr'] = cur_lr_d * d_lr_scale

            # R1+R2 gradient penalty (lazy: every r1r2_interval steps)
            r1_val = 0.0; r2_val = 0.0
            if d_step % args.r1r2_interval == 0:
                if args.r1_gamma > 0:
                    real_r1 = real.clone().detach().requires_grad_(True)
                    with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                        d_real_r1 = D(real_r1, real_labels)
                    r1_grad = torch.autograd.grad(outputs=d_real_r1.sum(), inputs=real_r1,
                                                   create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grad.pow(2).sum(dim=[1, 2, 3]).mean() * (args.r1_gamma / 2)
                    r1_penalty.backward()
                    r1_val = r1_penalty.item()
                if args.r2_gamma > 0:
                    fake_r2 = fake_d.clone().detach().requires_grad_(True)
                    with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                        d_fake_r2 = D(fake_r2, y_fake_d)
                    r2_grad = torch.autograd.grad(outputs=d_fake_r2.sum(), inputs=fake_r2,
                                                   create_graph=True, only_inputs=True)[0]
                    r2_penalty = r2_grad.pow(2).sum(dim=[1, 2, 3]).mean() * (args.r2_gamma / 2)
                    r2_penalty.backward()
                    r2_val = r2_penalty.item()

            for p in D.parameters():
                if p.grad is not None: torch.nan_to_num(p.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=p.grad)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 0.5)
            opt_D.step()
            d_loss_main = (d_loss_total / ga_d).item()
            torch.cuda.empty_cache()
        else:
            pass  # keep previous values

        # ADA update
        if args.ada and r_signs:
            batch_rt = torch.stack(r_signs).mean().item()
            ada_rt = ada_rt * 0.9 + batch_rt * 0.1
            adj = (d_batch * ga_d) / (args.ada_speed * 1000)
            if ada_rt > args.ada_target: ada_p += adj
            else: ada_p -= adj
            ada_p = max(args.ada_p_min, min(1.0, ada_p))

        # Progress
        g_nimg += g_nimg_per_step
        if d_ran:
            d_nimg += ga_d * d_batch; d_step += 1
        with torch.no_grad():
            if d_ran:
                r_acc = (d_real_last > 0).float().mean().item()
                f_acc = (d_fake_last < 0).float().mean().item()

        step += 1

        # ── Tick ──
        if step == 1 or (g_nimg - tick_start_nimg) >= args.tick_interval * 1000:
            tick_time = time.time(); elapsed = tick_time - t_start
            tick_dur = tick_time - tick_start_time
            g_cur_kimg = g_nimg / 1000
            sec_per_kimg = tick_dur / max(g_nimg - tick_start_nimg, 1) * 1000
            gpumem = torch.cuda.max_memory_allocated(device) / 2**30
            cpumem = psutil.Process(os.getpid()).memory_info().rss / 2**30
            eta_sec = sec_per_kimg * (total_kimg - g_cur_kimg)
            eta_str = f'{eta_sec/3600:.1f}h' if eta_sec > 3600 else f'{eta_sec/60:.0f}m'

            d_cur_kimg = d_nimg / 1000
            epoch = g_nimg / N
            msg = (f"tick {g_cur_kimg:5.0f}/{total_kimg}k G_kimg {g_cur_kimg:8.1f} D_kimg {d_cur_kimg:8.1f} ep {epoch:5.1f} "
                   f"time {dnnlib_format_time(elapsed):<12s} sec/tick {tick_dur:7.1f} sec/kimg {sec_per_kimg:7.2f} | "
                   f"g_gan={g_gan_main:.4f} g_mean={g_mean:+.3f} g_worst={g_worst:+.3f} g_best={g_best:+.3f} | "
                   f"d_loss={d_loss_main:.4f} d_gap={d_gap:+.3f} d_real={d_real_mean:+.3f} d_fake={d_fake_mean:+.3f} d_fake_max={d_fake_max:+.3f} | "
                   f"r_acc={r_acc:.3f} f_acc={f_acc:.3f} | "
                   f"adv+={adv_pos_ratio:.2f} adv_m={adv_mean_val:.3f} adv_s={adv_std_val:.3f} | "
                   f"ada_p={ada_p:.3f} ada_rt={ada_rt:.3f} d_lr_scale={d_lr_scale:.2f} r1={r1_val:.4f} r2={r2_val:.4f} | "
                   f"cpumem {cpumem:.2f}g gpumem {gpumem:.1f}g ga_g={ga_g} ga_d={ga_d} lr_g={cur_lr_g:.6f} lr_d={cur_lr_d*d_lr_scale:.6f} eta={eta_str}")
            print(msg)
            log_f.write(msg + '\n'); log_f.flush()

            if tb_writer is not None:
                for k, v in {'Loss/g_gan': g_gan_main, 'Loss/d_loss': d_loss_main,
                             'Loss/g_mean': g_mean, 'Loss/g_worst': g_worst, 'Loss/g_best': g_best,
                             'Balance/d_gap': d_gap, 'Balance/r_acc': r_acc, 'Balance/f_acc': f_acc,
                             'Group/adv_pos_ratio': adv_pos_ratio, 'Group/adv_mean': adv_mean_val, 'Group/adv_std': adv_std_val,
                             'ADA/ada_p': ada_p, 'ADA/ada_rt': ada_rt, 'Loss/r1': r1_val,
                             'Resources/gpu_mem_gb': gpumem, 'Timing/sec_per_kimg': sec_per_kimg}.items():
                    tb_writer.add_scalar(k, v, g_nimg)
                tb_writer.flush()

            # ── Snapshots ──
            if int(g_cur_kimg) % args.snap < args.tick_interval:
                # Swap to EMA weights for inference
                saved_state = None
                if G_ema is not None:
                    saved_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
                    G.load_state_dict(G_ema, strict=True)
                G.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=USE_BF16, dtype=torch.bfloat16):
                        fake_viz, colors_viz, masks_viz = G(fixed_z_viz, fixed_y_viz)
                    fake_viz = fake_viz.clamp(0, 1)

                    # All fakes grid (fixed_z + fixed_y)
                    n_viz = fake_viz.shape[0]
                    grid_cols = int(n_viz ** 0.5 + 0.5) if nc > 0 else 8
                    imgs = [to_np(fake_viz[i].cpu()) for i in range(n_viz)]
                    make_grid(imgs, grid_cols).save(img_dir / 'fakes' / f'fakes{int(g_nimg):09d}.png')

                    # Token decomposition
                    make_token_viz(fake_viz[:2], colors_viz[:2], masks_viz[:2], args.num_tokens).save(img_dir / 'tokens' / f'tokens{int(g_nimg):09d}.png')

                    # best_fake_g: G training batch top-16, 4×4 grid (already sorted in G phase)
                    top_n = 16
                    top_g = g_best_fakes[:top_n]
                    g_imgs = [to_np(img.clamp(0, 1)) for _, img in top_g]
                    black = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                    while len(g_imgs) < top_n:
                        g_imgs.append(black)
                    make_grid(g_imgs, 4).save(img_dir / 'best_fake_g' / f'best_fake_g{int(g_nimg):09d}.png')

                    # best_fake_d: D batch fake top-16, 4×4 grid (sort now, skip if D didn't run)
                    if d_ran:
                        d_best_fakes.sort(key=lambda x: x[0], reverse=True)
                        top_d = d_best_fakes[:top_n]
                        d_imgs = [to_np(img.clamp(0, 1)) for _, img in top_d]
                        while len(d_imgs) < top_n:
                            d_imgs.append(black)
                        make_grid(d_imgs, 4).save(img_dir / 'best_fake_d' / f'best_fake_d{int(g_nimg):09d}.png')
                # Restore training weights after EMA snapshot
                if saved_state is not None:
                    G.load_state_dict(saved_state, strict=True)
                    del saved_state
                G.train()

            # ── Checkpoint ──
            if int(g_cur_kimg) % args.ckpt_interval < args.tick_interval:
                ckpt_dict = {'G': G.state_dict(), 'D': D.state_dict(),
                            'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
                            'nimg': g_nimg, 'd_nimg': d_nimg, 'd_step': d_step,
                            'ada_p': ada_p, 'ada_rt': ada_rt}
                if G_ema is not None:
                    ckpt_dict['G_ema'] = G_ema
                torch.save(ckpt_dict, run_dir / f'ckpt_{int(g_cur_kimg):05d}k.pt')

            tick_start_time = time.time(); tick_start_nimg = g_nimg

    elapsed = time.time() - t_start
    final_dict = {'G': G.state_dict(), 'D': D.state_dict(),
                  'nimg': g_nimg, 'ada_p': ada_p, 'ada_rt': ada_rt}
    if G_ema is not None:
        final_dict['G_ema'] = G_ema
    torch.save(final_dict, run_dir / 'final.pt')
    log_f.close()
    if tb_writer is not None: tb_writer.close()
    print(f'[Done] total={elapsed:.0f}s')


if __name__ == '__main__':
    main()
