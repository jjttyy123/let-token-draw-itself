"""Calc FID for trained TokenPainter checkpoints. With detailed timing."""
import sys, os, torch, numpy as np, json, time
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models.inception import Inception_V3_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from TokenPainter.model import TokenPainter64

# ==== Configure =====================================================
# Folder of real images (PNG/JPG) used as the FID reference distribution.
MF = Path('./data/metfaces_64')
# One entry per model to evaluate: label -> run directory containing
# ckpt_*.pt / final.pt checkpoints written by train_gan_group.py.
# The integer after 'T=' is read as num_tokens.
RUNS = {
    'T=8': Path('./runs/T8'),
    'T=6': Path('./runs/T6'),
}
OUT = Path('./runs/fid_results')
OUT.mkdir(parents=True, exist_ok=True)
D = 'cuda'
N_FID = 1000
G_BS = 64   # generation batch
I_BS = 64   # inception batch

def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f'[{t}] {msg}'
    print(line, flush=True)
    with open(OUT / 'fid_log.txt', 'a') as f:
        f.write(line + '\n')

log('Loading InceptionV3...')
t0 = time.time()
inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
inception.fc = torch.nn.Identity()
inception = inception.to(D).eval()
log(f'Inception loaded in {time.time()-t0:.1f}s')

def get_acts(images, tag=''):
    """images: tensor (N,3,H,W) [0,1] → activations (N,2048)"""
    t0 = time.time(); N = len(images)
    all_a = []
    for i in range(0, N, I_BS):
        b = images[i:i+I_BS].to(D)
        b = torch.nn.functional.interpolate(b, size=299, mode='bilinear', align_corners=False)
        b = (b * 2) - 1
        with torch.no_grad(): a = inception(b)
        all_a.append(a.cpu()); del b
    result = torch.cat(all_a).numpy()
    log(f'  Inception: {N} imgs in {time.time()-t0:.1f}s ({I_BS}/batch)')
    return result

def sqrtm_sym(M):
    """sqrtm of symmetric PSD matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def calc_fid(a1, a2):
    t0 = time.time()
    m1, s1 = a1.mean(0), np.cov(a1, rowvar=False)
    m2, s2 = a2.mean(0), np.cov(a2, rowvar=False)
    log(f'  cov: {time.time()-t0:.1f}s')
    t0 = time.time()
    d = m1 - m2
    # s1@s2 is not symmetric, but s1^1/2 @ s2 @ s1^1/2 is symmetric
    # and Tr(sqrtm(s1@s2)) = Tr(sqrtm(s1^1/2 @ s2 @ s1^1/2))
    sqrt_s1 = sqrtm_sym(s1)
    M = sqrt_s1 @ s2 @ sqrt_s1
    cm_sqrt = sqrtm_sym(M)
    fid_val = float(d.dot(d) + np.trace(s1 + s2 - 2 * cm_sqrt))
    log(f'  FID done: {time.time()-t0:.1f}s')
    return max(fid_val, 0.0)

# Real activations
log('Loading real images...')
t0 = time.time()
real_imgs = []
for fn in sorted(os.listdir(MF)):
    if fn.lower().endswith(('.png','.jpg','.jpeg')):
        img = Image.open(MF / fn).convert('RGB').resize((64,64))
        real_imgs.append(transforms.ToTensor()(img))
        if len(real_imgs) >= N_FID: break
log(f'Loaded {len(real_imgs)} images in {time.time()-t0:.1f}s')
real_act = get_acts(torch.stack(real_imgs))
log(f'Real acts: {real_act.shape}')

results = {}
for name, rd in RUNS.items():
    log(f'\n=== {name} ===')
    Tok = int(name.split('=')[1])
    ckpts = sorted([p for p in rd.glob('ckpt_*.pt') if 'ckpt_00000k' not in p.name])
    ckpts.append(rd / 'final.pt')
    kvals, fids = [], []
    for cp in ckpts:
        kv = 'final' if cp.name == 'final.pt' else cp.stem.split('_')[1].replace('k','')
        log(f'  [{kv}] loading...')
        t0 = time.time()
        ckpt = torch.load(cp, map_location=D, weights_only=False)
        m = TokenPainter64(num_tokens=Tok).to(D).eval()
        m.load_state_dict(ckpt['G_ema'], strict=True)
        log(f'  [{kv}] model loaded in {time.time()-t0:.1f}s')

        t0 = time.time()
        gen_imgs, gen_count = [], 0
        with torch.no_grad():
            while gen_count < N_FID:
                B = min(G_BS, N_FID - gen_count)
                gen_imgs.append(m(torch.randn(B, 128, device=D))[0].cpu())
                gen_count += B
        gen_t = torch.cat(gen_imgs)[:N_FID]
        log(f'  [{kv}] gen {N_FID} imgs in {time.time()-t0:.1f}s ({G_BS}/batch)')

        t0 = time.time()
        gen_act = get_acts(gen_t)
        f = calc_fid(real_act, gen_act)
        kvals.append(kv); fids.append(round(f, 2))
        log(f'  [{kv}] FID={f:.1f} (total {time.time()-t0:.1f}s)')

        del m, ckpt, gen_t, gen_act
        torch.cuda.empty_cache()

    results[name] = {'kimg': kvals, 'fid': fids}
    # Save incrementally
    with open(OUT / 'fid.json', 'w') as f: json.dump(results, f)

log(f'\nAll done! Results: {OUT}/fid.json')

# Plot
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
for name, d in results.items():
    x = [float(v) if v != 'final' else 1000 for v in d['kimg']]
    plt.plot(x, d['fid'], 'o-', label=name, markersize=5)
plt.xlabel('kimg'); plt.ylabel('FID'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(OUT / f'fid.{ext}', dpi=150, bbox_inches='tight')
log('Plot saved')
