import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.jedi.edm import precond_coeffs, sample_sigma
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def make_windows(emb, act_emb, history):
    """Slice a (B, T, ...) sequence into all next-step prediction windows.

    Returns ctx (B*W, H, D), act_ctx (B*W, H, A), tgt (B*W, D) where
    W = T - H and target w is the frame right after context window w.
    """
    T = emb.size(1)
    ctx = emb[:, : T - 1].unfold(1, history, 1)  # (B, W, D, H)
    ctx = rearrange(ctx, 'b w d h -> (b w) h d')
    act_ctx = act_emb[:, : T - 1].unfold(1, history, 1)
    act_ctx = rearrange(act_ctx, 'b w a h -> (b w) h a')
    tgt = rearrange(emb[:, history:], 'b w d -> (b w) d')
    return ctx, act_ctx, tgt


def jedi_forward(self, batch, stage, cfg):
    """Encode observations, denoise noised future latents, compute JEDI loss.

    Loss (JEDI Eq., F-space EDM objective with stop-grad targets):
        || F_theta(c_in * z_noisy, c_noise, ctx, act)
           - (sg(z0) - c_skip * sg(z_noisy)) / c_out ||^2
    The encoder trains end-to-end through the *conditioning* path (ctx is not
    detached); stop-grad on the target side prevents representation collapse.
    With prob `diffusion.self_cond_prob` per batch the context is switched to
    the denoiser's own sampled rollout (paper's conditioning switch, reduces
    rollout exposure bias).
    """

    H = cfg.wm.history_size
    lambd = cfg.loss.sigreg.weight
    diff = cfg.diffusion

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    # Lance stores all tabular columns as float32; Embedding needs Long indices
    target = cfg.model.action_encoder.get('_target_', '')
    if 'Embedding' in target:
        batch['action'] = batch['action'].squeeze(-1).long()

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D), tanh-clamped
    act_emb = output['act_emb']

    ctx, act_ctx, tgt = make_windows(emb, act_emb, H)

    # JEDI conditioning switch: per batch, with uniform probability, condition
    # the denoiser on its own sampled rollout latents instead of encoder
    # latents (targets stay stop-grad encoder outputs). The first H frames
    # stay graph-connected encoder latents, so the encoder still receives
    # conditioning-path gradient on switched batches.
    self_cond = stage == 'fit' and torch.rand(()).item() < diff.self_cond_prob
    if self_cond:
        sampled = self.model.self_rollout(emb, act_emb, H)
        mixed = torch.cat([emb[:, :H], sampled], dim=1)
        ctx, _, _ = make_windows(mixed, act_emb, H)

    # noise the (stop-grad) targets at a log-normal noise level
    z0 = tgt.detach()
    sigma = sample_sigma(
        z0.size(0), diff.p_mean, diff.p_std, device=z0.device, dtype=z0.dtype
    )
    z_noisy = z0 + sigma * torch.randn_like(z0)

    c_skip, c_out, c_in, c_noise = precond_coeffs(sigma, diff.sigma_data)
    F_out = self.model.denoiser(c_in * z_noisy, c_noise, ctx, act_ctx)
    F_target = (z0 - c_skip * z_noisy) / c_out

    output['denoise_loss'] = (F_out - F_target).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['denoise_loss'] + lambd * output['sigreg_loss']

    batch_size = emb.size(0)
    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(
        losses_dict, on_step=True, sync_dist=True, batch_size=batch_size
    )

    # ---- diagnostics (not part of the loss) ----
    with torch.no_grad():
        # denoised estimate vs clean target, in latent space
        denoised = c_skip * z_noisy + c_out * F_out
        recon_mse = (denoised - z0).pow(2).mean()
        # persistence baseline: predict "no change" from the last context frame
        persist_mse = (ctx[:, -1] - z0).pow(2).mean()
        # per-dim std of embeddings across the batch (near 0 => collapse)
        emb_flat = emb.reshape(-1, emb.size(-1))  # (B*T, D)
        emb_std = emb_flat.std(dim=0).mean()
        diag = {
            f'{stage}/denoised_recon_mse': recon_mse,
            f'{stage}/persistence_baseline': persist_mse,
            f'{stage}/emb_std': emb_std,
            f'{stage}/sigma_mean': sigma.mean(),
            f'{stage}/self_cond': torch.tensor(
                float(self_cond), device=emb.device
            ),
        }

        # fixed-sigma probes: comparable across runs (random sigma is not)
        if self.global_step % 100 == 0:
            for probe_sigma in (0.1, 1.0):
                s = torch.full_like(sigma, probe_sigma)
                zs = z0 + s * torch.randn_like(z0)
                cs, co, ci, cn = precond_coeffs(s, diff.sigma_data)
                f_p = self.model.denoiser(ci * zs, cn, ctx, act_ctx)
                d_p = cs * zs + co * f_p
                diag[f'{stage}/recon_mse_sigma_{probe_sigma}'] = (
                    (d_p - z0).pow(2).mean()
                )

            # effective rank of the batch covariance. SIGReg is blind to
            # low-rank structure (linear projections of a low-rank Gaussian
            # are still Gaussian), so dimensional collapse must be tracked
            # directly: healthy runs should see this climb toward D.
            # autocast off: eigvalsh has no bf16 kernel and would also
            # re-downcast the .float() matmul back to bf16.
            with torch.autocast(device_type=emb.device.type, enabled=False):
                centered = (emb_flat - emb_flat.mean(dim=0)).float()
                cov = centered.T @ centered / (centered.size(0) - 1)
                evals = torch.linalg.eigvalsh(cov).clamp_min(0)
                p = evals / evals.sum().clamp_min(1e-12)
                entropy = -(p * (p + 1e-12).log()).sum()
                diag[f'{stage}/emb_effective_rank'] = entropy.exp()
    self.log_dict(diag, on_step=True, sync_dist=True, batch_size=batch_size)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='jedi_tennis')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        target = cfg.model.action_encoder.get('_target_', '')
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            if col == 'action' and 'Embedding' in target:
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        if 'Embedding' not in target:
            cfg.model.action_encoder.input_dim = (
                cfg.data.dataset.frameskip * dataset.get_dim('action')
            )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    scheduler = {
        'type': 'LinearWarmupCosineAnnealingLR',
        'warmup_steps': max(1, int(0.01 * total_steps)),
        'max_steps': total_steps,
    }
    # JEDI trains the encoder end-to-end at a fraction of the denoiser LR.
    # Regex order matters: encoder/projector must match before the catch-all.
    encoder_optimizer = dict(cfg.optimizer)
    encoder_optimizer['lr'] = cfg.optimizer.lr * cfg.encoder_lr_scale
    optimizers = {
        'encoder_opt': {
            'modules': r'model\.(encoder|projector)',
            'optimizer': encoder_optimizer,
            'scheduler': dict(scheduler),
            'interval': 'epoch',
        },
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': dict(scheduler),
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(jedi_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
