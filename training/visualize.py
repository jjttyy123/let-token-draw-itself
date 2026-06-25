"""Generate V5RL-v6 T=4 MSE experiment visualization images."""
import os, sys, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / '03_models'))
from pixel_rl import DecoderActorRL

RUN_DIR  = Path(__file__).resolve().parent / 'runs' / 'mps_overfit' / '20260604_135738' / 'T4_B32_lr0.0008_E100'
OUT_DIR  = Path(__file__).resolve().parent / 'MSE_T4_to_T8_images'
DEVICE   = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
Z_DIM    = 128
VIZ_IDX  = 1  # image index for token viz

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load data ───
fixed_z = torch.load(RUN_DIR / 'fixed_z.pt', map_location='cpu', weights_only=True)
z_all = fixed_z['z'].to(DEVICE)

TINYHERO_DIR = Path(__file__).resolve().parent.parent.parent / '02_dataset_and_dataset_research' / 'TinyHero' / '2'
imgs = []
for f in sorted(TINYHERO_DIR.glob('*.png')):
    arr = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
    imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
images = torch.stack(imgs).clamp(0, 1).to(DEVICE)
n_images = len(images)
print(f"Loaded {n_images} images")

# ─── Helpers ───
def to_np(t):
    t = t.detach().clamp(0, 1)
    if t.dim() == 3: t = t.permute(1, 2, 0)
    return t.cpu().mul(255).byte().numpy()

def make_grid(imgs_list, ncols, gap=2):
    h, w = imgs_list[0].shape[0], imgs_list[0].shape[1]
    nrows = (len(imgs_list) + ncols - 1) // ncols
    H = nrows * h + (nrows - 1) * gap
    W = ncols * w + (ncols - 1) * gap
    canvas = np.full((H, W, 3), 128, dtype=np.uint8)
    for i, img in enumerate(imgs_list):
        r, c = i // ncols, i % ncols
        canvas[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = img
    return Image.fromarray(canvas)

def load_model(ckpt_path, T=4):
    os.environ['H_HEAD'] = 'deeprgb_ln_mask32'
    os.environ['H_COMPOSE'] = 'softmax_bg'
    model = DecoderActorRL(z_dim=128, token_dim=512, num_tokens=T,
        num_layers=4, heads=4, mlp_ratio=8.0, dropout=0.1,
        bg_color=1.0, use_input_proj=True).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = ckpt['model'] if 'model' in ckpt else ckpt['model_state']
    model.load_state_dict(sd)
    model.eval()
    return model

def build_model_with_T(model_t4, T_new):
    os.environ['H_HEAD'] = 'deeprgb_ln_mask32'
    os.environ['H_COMPOSE'] = 'softmax_bg'
    model_new = DecoderActorRL(z_dim=128, token_dim=512, num_tokens=T_new,
        num_layers=4, heads=4, mlp_ratio=8.0, dropout=0.1,
        bg_color=1.0, use_input_proj=True).to(DEVICE)
    sd4 = model_t4.state_dict()
    sd_new = model_new.state_dict()
    for k in sd4:
        if k in sd_new:
            if sd4[k].shape == sd_new[k].shape:
                sd_new[k] = sd4[k]
            elif 'pos_embed' in k:
                n = min(sd4[k].shape[1], sd_new[k].shape[1])
                sd_new[k][:, :n, :] = sd4[k][:, :n, :]
    model_new.load_state_dict(sd_new)
    model_new.eval()
    return model_new

def mse(a, b):
    return float(((a - b) ** 2).mean())

def make_token_viz(gt_img, canvas, colors, masks, T, mse_val, bg_tensor):
    """Create token viz: Ref/Color/Mask/Accum rows with GT+Gen+Tokens columns."""
    gt_pil  = Image.fromarray((gt_img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
    gen_pil = Image.fromarray((canvas.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
    white   = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))

    vc, vm = colors[:T].clamp(0, 1), masks[:T].clamp(0, 1)

    # Every row: GT + Gen + tokens
    row0 = [gt_pil, gen_pil] + [white] * T  # Ref
    row1 = [gt_pil, gen_pil]  # Color
    for k in range(T):
        c = (vc[k].detach() * 255).byte().cpu().numpy()
        row1.append(Image.fromarray(np.full((64, 64, 3), c, dtype=np.uint8)))
    row2 = [gt_pil, gen_pil]  # Mask
    for k in range(T):
        row2.append(Image.fromarray((vm[k].unsqueeze(-1).expand(64, 64, 3) * 255).byte().cpu().numpy()))
    row3 = [gt_pil, gen_pil]  # Accum: tok_sum_k + (1-Σw)*bg, last k → Gen
    tok_sum = torch.zeros_like(bg_tensor)
    w_sum = torch.zeros(1, 1, 64, 64, device=bg_tensor.device)
    for k in range(T):
        w = vm[k:k+1]
        tok_sum = tok_sum + w * vc[k].view(3, 1, 1)
        w_sum = w_sum + w
        partial = tok_sum + (1 - w_sum) * bg_tensor
        row3.append(Image.fromarray((partial[0].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()))

    ncols = 2 + T
    cols = ['GT', 'Gen'] + [f'T{k}' for k in range(T)]
    H, W, gap, hh = 64, 64, 1, 14
    ow = ncols * (W + gap)
    oh = 4 * (H + gap) + hh + 30
    out = Image.new('RGB', (ow, oh), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        font = ImageFont.load_default()

    for ci, lbl in enumerate(cols):
        draw.text((ci * (W + gap) + 2, 1), lbl, fill=(0, 0, 0), font=font)
    for ri, lbl in enumerate(['Ref', 'Color', 'Mask', 'Accum']):
        draw.text((2, hh + ri * (H + gap) + 2), lbl, fill=(100, 100, 100), font=font)

    for i, img in enumerate(row0 + row1 + row2 + row3):
        ri, ci = i // ncols, i % ncols
        out.paste(img, (ci * (W + gap), hh + ri * (H + gap) + 2))

    return out

# ─── Load models ───
model_t4 = load_model(RUN_DIR / 'final.pt', T=4)
model_t2 = build_model_with_T(model_t4, 2)
model_t6 = build_model_with_T(model_t4, 6)

# ═══════════════════════════════════════════════════════
# 1. Compare: 8 random images
# ═══════════════════════════════════════════════════════
print("[1/4] Reconstruction progress: same z across epochs...")
g = torch.Generator(device='cpu').manual_seed(42)
idx8 = torch.randperm(n_images, generator=g)[:8]
gt8_np = [to_np(images[i]) for i in idx8]
z8 = z_all[idx8]

all_rows = [gt8_np]  # Row 0: GT
for ep, ckpt_name in [(25, 'E25'), (50, 'E50'), (75, 'E75'), (100, 'E100')]:
    m = load_model(RUN_DIR / f'ckpt_e{ep:05d}.pt', T=4)
    with torch.no_grad():
        gen, _, _ = m(z8)
    all_rows.append([to_np(gen[i].cpu()) for i in range(8)])
    print(f"  {ckpt_name} done")

grid = make_grid(sum(all_rows, []), 8)  # 5 rows × 8 cols
W, H = grid.size
labeled = Image.new('RGB', (W + 30, H), (255, 255, 255))
labeled.paste(grid, (30, 0))
draw = ImageDraw.Draw(labeled)
try:
    font14 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except:
    font14 = ImageFont.load_default()
for ri, lbl in enumerate(['GT', 'E25', 'E50', 'E75', 'E100']):
    y = ri * (64 + 2) + 64 // 2 - 8
    draw.text((2, y), lbl, fill=(80, 80, 80), font=font14)
labeled.save(OUT_DIR / '01_recon_progress.png')
print(f"  -> {OUT_DIR / '01_recon_progress.png'}")

# ═══════════════════════════════════════════════════════
# 2. Token viz: T=2, T=4, T=6
# ═══════════════════════════════════════════════════════
print("[2/4] Token viz...")
model_t8  = build_model_with_T(model_t4, 8)
model_t12 = build_model_with_T(model_t4, 12)
model_t16 = build_model_with_T(model_t4, 16)
gt_img = images[VIZ_IDX]
z_fix = z_all[VIZ_IDX:VIZ_IDX+1]

def build_panels(t_list, model_map):
    panels = []
    for T in t_list:
        m = model_map[T]
        with torch.no_grad():
            canvas, colors, masks = m(z_fix)
        mse_val = mse(canvas[0], gt_img)
        title = f"T={T}  MSE={mse_val:.6f}"
        print(f"  {title}")
        panel = make_token_viz(gt_img, canvas[0], colors[0], masks[0], T, mse_val, m.bg.expand(1, -1, -1, -1))
        panels.append((title, panel))
    return panels

def save_combined(panels, out_name):
    max_w = max(p.width for _, p in panels)
    total_h = sum(p.height for _, p in panels) + 30
    combined = Image.new('RGB', (max_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(combined)
    try:
        lbl_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        lbl_font = ImageFont.load_default()
    y = 0
    for title, panel in panels:
        draw.text((2, y + 2), title, fill=(200, 0, 0), font=lbl_font)
        combined.paste(panel, (0, y + 18))
        y += panel.height + 20
    combined.save(OUT_DIR / out_name)
    print(f"  -> {OUT_DIR / out_name}")

model_map = {2: model_t2, 4: model_t4, 6: model_t6, 8: model_t8, 12: model_t12, 16: model_t16}

save_combined(build_panels([2, 4, 6], model_map), '02a_token_viz_T2_T4_T6.png')
save_combined(build_panels([8, 12, 16], model_map), '02b_token_viz_T8_T12_T16.png')

# ═══════════════════════════════════════════════════════
# 3. 4 checkpoints × 4 random z
# ═══════════════════════════════════════════════════════
print("[3/4] 4 ckpt x 8 random z...")
g = torch.Generator(device='cpu').manual_seed(777)
z_rand = torch.randn(8, Z_DIM, generator=g).to(DEVICE)

all_gen = []
for ep in [25, 50, 75, 100]:
    m = load_model(RUN_DIR / f'ckpt_e{ep:05d}.pt', T=4)
    with torch.no_grad():
        canvas, _, _ = m(z_rand)
    for i in range(8):
        all_gen.append(to_np(canvas[i].cpu()))

grid = make_grid(all_gen, 8)
W, H = grid.size
labeled = Image.new('RGB', (W + 30, H), (255, 255, 255))
labeled.paste(grid, (30, 0))
draw = ImageDraw.Draw(labeled)
try:
    font11 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
except:
    font11 = ImageFont.load_default()
for ri, ep in enumerate([25, 50, 75, 100]):
    y = ri * (64 + 2) + 64 // 2 - 7
    draw.text((2, y), f"E{ep}", fill=(0, 0, 0), font=font11)
labeled.save(OUT_DIR / '03_ckpt_random_z.png')
print(f"  -> {OUT_DIR / '03_ckpt_random_z.png'}")

# ═══════════════════════════════════════════════════════
# 4. Training curve
# ═══════════════════════════════════════════════════════
print("[4/4] Training curve...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_lines = open(RUN_DIR / 'train.log').readlines()
epochs, losses, bests = [], [], []
for line in log_lines:
    line = line.strip()
    if not line.startswith('E'): continue
    parts = line.split()
    loss_val = best_val = None
    for p in parts:
        if p.startswith('loss='): loss_val = float(p.split('=')[1])
        elif p.startswith('best='): best_val = float(p.split('=')[1])
    if loss_val is not None and best_val is not None:
        epochs.append(int(parts[1]))
        losses.append(loss_val)
        bests.append(best_val)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, losses, 'b-', alpha=0.5, linewidth=0.5, label='Train Loss')
ax.plot(epochs, bests, 'r-', linewidth=1.5, label='Best Loss')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE (log scale)')
ax.set_title('V5RL-v6 T=4 MSE Overfit on TinyHero (Front, 912 images)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / '04_training_curve.png', dpi=150); plt.close()

print(f"\nDone! => {OUT_DIR}")
for f in sorted(OUT_DIR.glob('*.png')):
    print(f"  {f.name}")
