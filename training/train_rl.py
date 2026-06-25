"""V5 GRPO — Co-training G+D, group-normalized G loss.

D step: standard hinge loss (same as GAN)
G step: sample K per z → score → group normalize → advantage-weighted loss

Gradient flows through D to G (like GAN G-step), D sees real/fake.
"""
import os, sys, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../03_models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pixel_rl import DecoderActor, DecoderActorRL, ViTRewardModel
from pixel_rl.token_d import TokenD
from pretrain_config import (
    CANVAS_SIZE, PALETTE_SIZE,
    ACTOR_HEADS, ACTOR_MLP_RATIO, ACTOR_DROPOUT,
    Z_DIM, BATCH_SIZE, ACTOR_LR,
    DATA_DIR, SAVE_INTERVAL, EVAL_INTERVAL, LOG_INTERVAL,
    ACTOR_TOKEN_DIM, ACTOR_NUM_LAYERS,
    R1_GAMMA,
)


class SAGANDWrapper(nn.Module):
    """Original SAGAN D, no modification."""
    def __init__(self, df_dim=64):
        super().__init__()
        from sagan import UnconditionalDiscriminator
        self.d = UnconditionalDiscriminator(df_dim=df_dim)

    def forward(self, x):
        s = self.d(x * 2.0 - 1.0)  # [0,1] → [-1,1]
        return s, torch.zeros(x.shape[0], 512, device=x.device)

    def score(self, x):
        return self.d(x * 2.0 - 1.0).squeeze(-1)


class TokenDWrapper(nn.Module):
    def __init__(self, mode='invdec', cnn_dim=32, token_dim=128,
                 num_queries=8, num_layers=4, num_heads=8, use_sn=False):
        super().__init__()
        self.d = TokenD(mode=mode, cnn_dim=cnn_dim, token_dim=token_dim,
                        num_queries=num_queries, num_layers=num_layers,
                        num_heads=num_heads, use_sn=use_sn)

    def forward(self, x):
        return self.d(x)

    def score(self, x):
        return self.d.score(x)


def train():
    # ── DDP init ──
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    is_main = (local_rank == 0)
    torch.backends.cudnn.benchmark = True

    # ── Config via env ──
    _g_dim = int(os.environ.get('H_TOKEN_DIM', '512'))
    _g_layers = int(os.environ.get('H_NUM_LAYERS', '4'))
    _d_type = os.environ.get('H_GRPO_D_TYPE', 'invdec')
    _d_cnn = int(os.environ.get('H_TD_CNN_DIM', '32'))
    _d_dim = int(os.environ.get('H_TD_TOKEN_DIM', '128'))
    _sagan_df = int(os.environ.get('H_SAGAN_DF_DIM', '64'))
    _grpo_k = int(os.environ.get('H_GRPO_K', '8'))
    _epochs = int(os.environ.get('H_ViT_Epochs', '1000'))
    _g_lr = float(os.environ.get('H_ACTOR_LR', '0.0001'))
    _d_lr = float(os.environ.get('H_VIT_LR', '0.0004'))
    _freq = int(os.environ.get('H_VIT_UPDATE_FREQ', '1'))
    _seed = int(os.environ.get('H_SEED', '42'))
    _loss_type = os.environ.get('H_LOSS_TYPE', 'hinge_r1')
    _r1 = float(os.environ.get('H_R1_GAMMA', '10'))
    _rl_method = os.environ.get('H_RL_METHOD', 'group')  # group | ddpg | ranking
    _actor_type = os.environ.get('H_ACTOR_TYPE', 'v5b')  # v5b | rl

    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # ── Data (random sampling with replacement, like GAN) ──
    data_path = os.environ.get('H_DATA_DIR', os.path.join(DATA_DIR, 'tinyhero/class1'))
    data_file = os.environ.get('H_DATA_FILE', 'dataset_cache_mse.pt')
    real_data = torch.load(os.path.join(data_path, data_file), map_location='cpu', weights_only=True)
    N_data = len(real_data)

    # ── Actor ──
    actor_kwargs = dict(z_dim=Z_DIM, token_dim=_g_dim, num_layers=_g_layers,
                        num_tokens=PALETTE_SIZE, heads=ACTOR_HEADS,
                        mlp_ratio=ACTOR_MLP_RATIO, dropout=ACTOR_DROPOUT)
    if _actor_type == 'rl':
        actor = DecoderActorRL(**actor_kwargs, use_input_proj=True).to(device)
        actor_label = f"V5-RL"
    else:
        actor = DecoderActor(**actor_kwargs).to(device)
        actor_label = f"V5-B"
    if is_main:
        print(f"[Actor] {actor_label} dim={_g_dim} L={_g_layers} params={sum(p.numel() for p in actor.parameters()):,}")

    # ── Discriminator ──
    if _d_type == 'sagan':
        vit = SAGANDWrapper(df_dim=_sagan_df).to(device)
        d_label = f"SAGAN df={_sagan_df}"
    else:
        vit = TokenDWrapper(mode='invdec', cnn_dim=_d_cnn, token_dim=_d_dim,
                            num_queries=PALETTE_SIZE, num_layers=_g_layers,
                            num_heads=ACTOR_HEADS, use_sn=False).to(device)
        d_label = f"InvDecD cnn={_d_cnn} dim={_d_dim}"
    print(f"[D] {d_label} params={sum(p.numel() for p in vit.parameters()):,}")

    # ── DDP wrap ──
    if world_size > 1:
        actor = DDP(actor, device_ids=[local_rank])
        vit = DDP(vit, device_ids=[local_rank])
        actor_no_ddp = actor.module
        vit_no_ddp = vit.module
    else:
        actor_no_ddp = actor
        vit_no_ddp = vit

    opt_actor = torch.optim.Adam(actor.parameters(), lr=_g_lr, betas=(0.0, 0.9))
    opt_vit = torch.optim.Adam(vit.parameters(), lr=_d_lr, betas=(0.0, 0.9))

    # ── Resume from checkpoint ──
    _resume = os.environ.get('H_RESUME', '')
    resume_step = 0
    if _resume:
        print(f"[Resume] Loading {_resume}")
        ckpt = torch.load(_resume, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt['actor'], strict=False)
        if 'vit' in ckpt:
            vit.load_state_dict(ckpt['vit'], strict=False)
        resume_step = ckpt.get('step', 0)
        print(f"[Resume] Loaded step={resume_step}")

    # ── Logging ──
    run_name = os.environ.get('RUN_NAME', f'grpo_G{_g_dim}L{_g_layers}_{_d_type}_cnn{_d_cnn}_K{_grpo_k}')
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', run_name)
    samples_dir = os.path.join(log_dir, 'samples')
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))

    B = BATCH_SIZE
    K = _grpo_k
    total_steps = _epochs * (N_data // (B * world_size) + 1)
    total_steps = min(total_steps, int(os.environ.get('H_MAX_UPDATES', '999999')))
    eval_z = torch.randn(25, Z_DIM, device=device) * 1.2
    step = resume_step
    total_steps = total_steps + resume_step  # continue from checkpoint
    t0 = time.time()

    print(f"  RL_METHOD={_rl_method} K={K} loss={_loss_type} R1={_r1} FREQ={_freq} total_steps={total_steps} B={B}")

    ref_real = real_data[:4].to(device)

    @torch.no_grad()
    def generate_samples(step):
        actor.eval()
        imgs = actor_no_ddp.generate(eval_z)
        save_image(imgs, os.path.join(samples_dir, f'sample_{step:05d}.png'), nrow=5)
        # Token viz every 2000 steps
        if step % 2000 == 0:
            _gen_token_viz(eval_z[:4], ref_real, step)
        actor.train()

    @torch.no_grad()
    def _gen_token_viz(z_batch, real_batch, step):
        from PIL import Image, ImageDraw, ImageFont
        canvas, colors, masks = actor_no_ddp.sample(z_batch, tau=0.1)
        B = canvas.shape[0]
        white_bg = torch.ones(3, 64, 64, device=canvas.device)
        labels = ['GT', 'Gen', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
        NCOL = 10
        rows = []
        for b in range(B):
            real_img = real_batch[b].clamp(0, 1)
            row_a = [real_img, canvas[b].clamp(0, 1)] + [colors[b,k].view(3,1,1).expand(3,64,64).clamp(0,1) for k in range(8)]
            row_b = [real_img, white_bg.clone()] + [(masks[b,k:k+1]*colors[b,k].view(3,1,1)).clamp(0,1) for k in range(8)]
            accum = white_bg.clone()
            row_c = [real_img, canvas[b].clamp(0,1)]
            for k in range(8):
                mk, ck = masks[b,k:k+1], colors[b,k].view(3,1,1)
                accum = mk*ck+(1-mk)*accum
                row_c.append(accum.clamp(0,1))
            rows.extend(row_a + row_b + row_c)
        grid = torch.stack(rows)
        grid_np = (grid.permute(0,2,3,1).clamp(0,1)*255).byte().cpu().numpy()
        H, W = 64, 64
        hh = 16
        out_h = B*3*(H+2)+hh
        out_w = NCOL*(W+2)
        out_img = Image.new('RGB',(out_w,out_h),(255,255,255))
        draw = ImageDraw.Draw(out_img)
        try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",10)
        except: font = ImageFont.load_default()
        for ci,lbl in enumerate(labels):
            draw.text((ci*(W+2)+2,1),lbl,fill=(0,0,0),font=font)
        for i,img_np in enumerate(grid_np):
            ri,ci = (i//NCOL)%3,i%NCOL
            out_img.paste(Image.fromarray(img_np),(ci*(W+2)+2,hh+ri*(H+2)+2))
        out_img.save(os.path.join(samples_dir,f'tokens_{step:05d}.png'))

    while step < total_steps:
        # Random sampling with replacement — GAN-style
        idx = torch.randint(0, N_data, (B,))
        real = real_data[idx].to(device)

        # ═══════════════════════════════════════════════════════
        #  Actor forward + score
        # ═══════════════════════════════════════════════════════
        t_start = time.time()
        actor.train()
        NK = B * K
        z = torch.randn(NK, Z_DIM, device=device)
        canvas, _, _ = actor(z)
        scores = vit_no_ddp.score(canvas)          # DDP doesn't forward custom methods
        scores_grp = scores.reshape(B, K)

        # ── Actor loss ──
        if _rl_method == 'group':
            mean_s = scores_grp.mean(dim=1, keepdim=True)
            std_s = scores_grp.std(dim=1, keepdim=True) + 1e-8
            advantages = (scores_grp - mean_s) / std_s
            loss_actor = -(advantages.detach().reshape(-1) * scores.reshape(-1)).mean()
        elif _rl_method == 'ddpg':
            loss_actor = -scores.mean()
        else:
            raise ValueError(f"Unknown RL method: {_rl_method}")

        # D loss (best_fake detached)
        final_grp = scores_grp
        if step % _freq == 0:
            best_idx = final_grp.argmax(dim=1)
            best_fake = canvas.reshape(B, K, 3, 64, 64)[torch.arange(B, device=device), best_idx].detach()

            r1_penalty = torch.tensor(0.0, device=device)
            if _loss_type == 'hinge_r1':
                real.requires_grad_(True)
                r_logits, _ = vit(real)
                r1_grad = torch.autograd.grad(outputs=r_logits.sum(), inputs=real,
                                              create_graph=True, retain_graph=True)[0]
                r1_penalty = _r1 * 0.5 * r1_grad.reshape(B, -1).pow(2).sum(dim=1).mean()

            real_logits, _ = vit(real)
            fake_logits, _ = vit(best_fake)
            vit_loss = (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())
            vit_loss = vit_loss + r1_penalty

            d_real = real_logits.mean().item()
            d_fake = fake_logits.mean().item()
            d_gap = d_real - d_fake
            l_d = vit_loss.item()
            r_acc = (real_logits > 0).float().mean().item()
            f_acc = (fake_logits < 0).float().mean().item()
        else:
            vit_loss = torch.tensor(0.0)
            d_real = d_fake = d_gap = l_d = r_acc = f_acc = 0.0

        # ═══════════════════════════════════════════════════════
        #  Update Actor
        # ═══════════════════════════════════════════════════════
        opt_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        opt_actor.step()

        # ═══════════════════════════════════════════════════════
        #  Update D
        # ═══════════════════════════════════════════════════════
        if step % _freq == 0:
            opt_vit.zero_grad()
            vit_loss.backward()
            torch.nn.utils.clip_grad_norm_(vit.parameters(), 0.5)
            opt_vit.step()

        # ── Log ──
        if step % LOG_INTERVAL == 0:
            adv_std = final_grp.std(dim=1).mean().item()
            dt = time.time() - t_start
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (total_steps - step - 1) if step > 0 else 0
            print(f"\r  {step:6d}/{total_steps} [{eta/60:4.0f}m ETA {dt:.1f}s/step] "
                  f"D_real={d_real:7.3f} D_fake={d_fake:7.3f} D_gap={d_gap:7.3f}  "
                  f"r_acc={r_acc:.3f} f_acc={f_acc:.3f}  "
                  f"L_D={l_d:7.3f} L_G={loss_actor.item():7.3f}  "
                  f"adv_std={adv_std:.3f}",
                  end='', flush=True)

            if step % _freq == 0:
                writer.add_scalar('Loss/D', l_d, step)
                writer.add_scalar('Loss/G', loss_actor.item(), step)
                writer.add_scalar('Metrics/D_gap', d_gap, step)
                writer.add_scalar('Metrics/adv_std', adv_std, step)

        if step % EVAL_INTERVAL == 0:
            generate_samples(step)

        if step % SAVE_INTERVAL == 0 and step > 0:
            torch.save({'actor': actor.state_dict(), 'vit': vit.state_dict(),
                        'opt_actor': opt_actor.state_dict(), 'opt_vit': opt_vit.state_dict(),
                        'step': step},
                       os.path.join(ckpt_dir, f'ckpt_{step:06d}.pt'))

        step += 1

    torch.save({'actor': actor.state_dict(), 'vit': vit.state_dict(), 'step': step},
               os.path.join(log_dir, 'final.pt'))
    # Also save to checkpoints/ for Phase 2
    torch.save({'actor': actor.state_dict(), 'vit': vit.state_dict(), 'step': step},
               os.path.join(ckpt_dir, 'ckpt_final.pt'))
    generate_samples(total_steps)
    print(f"\n[Done] Steps={step}")
    writer.close()


if __name__ == '__main__':
    train()
