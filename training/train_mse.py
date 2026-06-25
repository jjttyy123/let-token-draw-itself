"""MSE overfit: single V5-RL variant, TinyHero flat images.
Fixed noise-image pairs — each z maps to a fixed target throughout training.
Env: H_HEAD, H_ATTN, H_VARIANT, H_BATCH_SIZE, H_ViT_Epochs, H_DATA_FILE
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../03_models'))
from pixel_rl import DecoderActorRL

Z_DIM, DIM, LAYERS = 128, 512, 4
T, CTX = 32, 8
LR, SEED, NUM_EVAL = 5e-4, 42, 4


@torch.no_grad()
def save_token_viz(model, z, path):
    _, colors, masks = model(z)
    bg = model.bg.expand(1, -1, -1, -1)
    Tpv, NCOL = CTX, CTX + 1
    BLACK = torch.zeros(3, 64, 64, device=z.device)
    rows = []
    for v in range(4):
        vs = v * Tpv
        vc, vm = colors[0, vs:vs+Tpv], masks[0, vs:vs+Tpv]
        cv = bg.clone()
        for k in range(Tpv):
            cv = vm[k:k+1] * vc[k].view(3,1,1) + (1 - vm[k:k+1]) * cv
        gen = cv[0].clamp(0, 1)
        rows.extend([gen] + [vc[k].view(3,1,1).expand(3,64,64).clamp(0,1) for k in range(Tpv)])
        rows.extend([BLACK.clone()] + [vm[k:k+1].expand(3,64,64).clamp(0,1) for k in range(Tpv)])
        acc = bg.clone()
        rows.extend([gen])
        for k in range(Tpv):
            acc = vm[k:k+1]*vc[k].view(3,1,1)+(1-vm[k:k+1])*acc
            rows.append(acc[0].clamp(0,1))
    grid = torch.stack(rows)
    grid_np = (grid.permute(0,2,3,1).clamp(0,1)*255).byte().cpu().numpy()
    H, W, gap, hh = 64, 64, 1, 14
    oh = 12 * (H+gap) + hh + 4
    ow = NCOL * (W+gap)
    out = Image.new('RGB', (ow, oh), (255,255,255))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font = ImageFont.load_default()
    for ci, lbl in enumerate(['Gen']+[f'T{k}' for k in range(Tpv)]):
        draw.text((ci*(W+gap)+2, 1), lbl, fill=(0,0,0), font=font)
    vn = ['Back','Left','Front','Right']
    rt = ['Color','Mask','Accum']
    for i, img_np in enumerate(grid_np):
        ri, ci = i // NCOL, i % NCOL
        vi, rti = ri // 3, ri % 3
        draw.text((ci*(W+gap)+W+2, hh+ri*(H+gap)+2), f'{vn[vi]} {rt[rti]}' if ci==0 else '',
                  fill=(100,100,100), font=font)
        out.paste(Image.fromarray(img_np), (ci*(W+gap), hh+ri*(H+gap)+2))
    out.save(path)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)

    # Config
    name = os.environ.get('H_VARIANT', 'v5RL_A')
    head = os.environ.get('H_HEAD', 'base')
    attn = os.environ.get('H_ATTN', 'standard')
    N_overfit = int(os.environ.get('H_BATCH_SIZE', '128'))
    epochs = int(os.environ.get('H_ViT_Epochs', '100'))

    # Data
    data_path = os.environ.get('H_DATA_FILE',
        os.path.join(os.path.dirname(__file__),
        '../../../02_dataset_and_dataset_research/pretrain_data/tinyhero/tinyhero_all_flat.pt'))
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    N_total = len(data)

    # Select N_overfit images as fixed overfit set
    torch.manual_seed(SEED)
    indices = torch.randperm(N_total)[:N_overfit]
    targets = data[indices]  # [N_overfit, 3, 64, 64]
    # Fixed noise paired with each target
    z_fixed = torch.randn(N_overfit, Z_DIM) * 1.2

    # Output dir
    run_name = os.environ.get('RUN_NAME', f'mse_{name}')
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', run_name)
    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    print(f"[{name}] head={head} attn={attn} N_overfit={N_overfit} epochs={epochs}")
    print(f"  Total images={N_total}, fixed pairs={N_overfit}")

    # Fixed eval z (same 4 vectors for consistent comparison)
    torch.manual_seed(SEED + 1)
    eval_z = torch.randn(NUM_EVAL, Z_DIM, device=device) * 1.2
    eval_gt = targets[torch.randint(0, N_overfit, (NUM_EVAL,))]

    # Model
    os.environ['H_HEAD'] = head
    os.environ['H_ATTN'] = attn
    model = DecoderActorRL(
        z_dim=Z_DIM, token_dim=DIM, num_tokens=T,
        num_layers=LAYERS, heads=8, mlp_ratio=4.0, dropout=0.1,
        bg_color=1.0, use_input_proj=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    best_loss = float('inf')
    steps_per_epoch = N_overfit // 128  # micro-batch size = 128 (GPU-friendly)

    for epoch in range(epochs):
        model.train()
        # Shuffle pair order each epoch
        perm = torch.randperm(N_overfit)
        epoch_losses = []

        for mb_start in range(0, N_overfit, 128):
            mb_idx = perm[mb_start:mb_start + 128]
            z_batch = z_fixed[mb_idx].to(device)
            target_batch = targets[mb_idx].to(device)

            canvas, _, _ = model(z_batch)
            loss = F.mse_loss(canvas, target_batch)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        best_loss = min(best_loss, avg_loss)

        if epoch % 10 == 0 or epoch < 5:
            elapsed = time.time() - t0
            eta = elapsed / (epoch + 1) * (epochs - epoch) if epoch > 0 else 0
            print(f"  epoch {epoch:3d}/{epochs} loss={avg_loss:.6f} best={best_loss:.6f} eta={eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"  DONE best={best_loss:.6f} time={elapsed:.1f}s")

    # Eval
    model.eval()
    with torch.no_grad():
        eval_canvas, _, _ = model(eval_z)
    compare = torch.cat([eval_gt.clamp(0,1), eval_canvas.cpu().clamp(0,1)], dim=0)
    save_image(compare, os.path.join(samples_dir, 'compare.png'), nrow=NUM_EVAL)
    save_token_viz(model, eval_z[:1], os.path.join(samples_dir, 'tokens.png'))
    torch.save({'model': model.state_dict(), 'loss': best_loss},
               os.path.join(log_dir, 'final.pt'))

    print(f"[Done] {log_dir}")


if __name__ == '__main__':
    train()
