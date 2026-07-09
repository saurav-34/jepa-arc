# JEDI integration plan

Integrating **JEDI — Joint Embedding Diffusion world model** (arXiv:2605.13013)
into the LeWM JEPA stack, on branch `jedi-diffusion-wm`.

## What JEDI changes vs. our current LeWM

| | LeWM (current) | JEDI |
|---|---|---|
| Predictor | deterministic AdaLN transformer, next-latent MSE | EDM-preconditioned diffusion denoiser over next latent |
| Anti-collapse | SIGReg on embeddings | stop-grad on future latent targets (denoising loss itself) |
| Encoder training | end-to-end through MSE | end-to-end through denoising loss, at 0.3× the denoiser LR |
| Rollout | one forward pass per step | few-step denoising sampler per step (stochastic) |
| Latents | unbounded | clamped with `C(z) = s * tanh(z/s)`, s = 3 |

Why this may help us: the deterministic MSE predictor regresses to the mean of
the next-state distribution, which is exactly the "ball_y softens in rollout"
blur we measured with the probes. A diffusion predictor can commit to one mode
per sample instead of averaging them.

Everything downstream (solvers, `get_cost`, probes, `save_pretrained`) keys off
the `encode / predict / rollout` interface, so JEDI is added as a sibling of
`wm/lewm/`, not a rewrite.

## Step 1 — EDM utilities (`stable_worldmodel/wm/jedi/edm.py`)

Pure functions, unit-testable in isolation:

- `sample_sigma(batch, P_mean, P_std)` — log-normal noise-level sampling.
- `precond_coeffs(sigma, sigma_data)` — returns `c_skip`, `c_out`, `c_in`,
  `c_noise` (Karras EDM formulas). With tanh-clamped latents at s = 3,
  `sigma_data ≈ 1.0–1.5` (measure empirically from a trained-encoder batch;
  start at 1.0).
- `NoiseEmbedding` — Fourier features of `c_noise` → small MLP, output dim =
  `embed_dim` so it can be summed into the AdaLN conditioning vector.
- `denoise(F, z_noisy, sigma, cond)` — wraps the raw network:
  `D = c_skip * z_noisy + c_out * F(c_in * z_noisy, c_noise, cond)`.
- `sample(F, cond, n_steps, sigma_max)` — few-step Euler sampler (paper /
  DIAMOND regime: ~3 steps). This is the inference-time `predict`.

## Step 2 — Denoiser network (`stable_worldmodel/wm/jedi/module.py`)

Reuse `Transformer` + `ConditionalBlock` from `wm/lewm/module.py` (import, do
not copy). Token layout per training example:

```
tokens: [z_{t-H+1} ... z_t,  z_noisy_{t+1}]     causal attention
cond c: act_emb per position  +  noise_emb (broadcast)
```

i.e. the same causal predictor shape as today, but the last position is the
*noised next latent* and the AdaLN conditioning gains a noise-level term.
Output at the last position is `F_theta(...)`; earlier positions give
next-step predictions for the intermediate frames (same trick as the current
`predict`, keeps per-step supervision dense).

`num_frames = history_size + 1` for the positional embedding.

## Step 3 — World model class (`stable_worldmodel/wm/jedi/jedi.py`)

`class JEDI(nn.Module)` mirroring the LeWM public API so the solver / policy /
probe stack works unchanged:

- `encode(info)` — identical to LeWM (ViT CLS → projector) **plus latent clamp**
  `3 * tanh(z / 3)` after the projector.
- `predict(emb, act_emb, z_noisy, sigma)` — training-time single denoiser call
  (returns raw `F` output, loss assembled outside).
- `sample_next(emb, act_emb, n_steps=3)` — inference: run the Euler sampler to
  produce `z_{t+1}`.
- `rollout(info, action_sequence, history_size)` — same autoregressive loop as
  `LeWM.rollout`, with `self.predict(...)[:, -1]` replaced by
  `self.sample_next(...)`. The S plan-samples dimension now doubles as the
  diffusion-sample dimension for free (each candidate gets its own stochastic
  rollout).
- `criterion` / `get_cost` — unchanged from LeWM (goal-latent MSE at last step).

Drop `pred_proj` (the denoiser output *is* latent-space); keep `projector`.

## Step 4 — Training forward + script (`scripts/train/jedi.py`)

Clone `scripts/train/lewm.py`, replace `lejepa_forward` with `jedi_forward`:

1. `output = model.encode(batch)` → clean latents `z` (B, T, D), clamped.
2. Sample `sigma` per sequence element; noise the *target* latents:
   `z_noisy = sg(z_{t+1}) + sigma * eps`.
3. Denoiser forward conditioned on context latents (NO stop-grad — this is the
   path that trains the encoder) + action embs + noise emb.
4. **JEDI loss** (paper Eq., stop-grad on targets only):
   `|| F_theta - (sg(z_{t+1}) - c_skip * sg(z_noisy)) / c_out ||^2`
5. Keep SIGReg as an optional regularizer behind a config weight
   (`loss.sigreg.weight`, default small e.g. 0.01, ablate to 0 later). JEDI's
   sg-targets should prevent collapse on their own, but we already know SIGReg
   works on this data — remove it only after the `emb_std` diagnostic stays
   healthy without it.
6. Keep the existing diagnostics block (copy baseline, `emb_std`), and add
   `denoise_loss` at 2–3 fixed sigmas so the curve is comparable across runs.

Optimizers: two param groups via the existing `optim` dict —
`encoder+projector` at `0.3 * lr`, `denoiser+action_encoder+noise_emb` at `lr`
(paper's encoder-LR ratio).

## Step 5 — Config (`scripts/train/config/jedi_tennis.yaml`)

Copy `lewm_tennis.yaml`; changes:

- `model._target_: stable_worldmodel.wm.jedi.JEDI`, predictor block → denoiser
  block (`num_frames: ${wm.history_size} + 1`).
- New `diffusion:` group: `sigma_data: 1.0`, `P_mean: -0.4`, `P_std: 1.2`
  (EDM defaults, tune later), `sample_steps: 3`, `latent_clamp: 3.0`.
- `optimizer.encoder_lr_scale: 0.3`.
- wandb project `tennis-jepa`, run name `jedi_tennis`.

## Step 6 — Verify before scaling

1. **Unit tests** (`tests/`): precond coeffs vs. EDM reference values; sampler
   with a known toy denoiser; loss is finite and decreases on one overfit batch.
2. **Overfit run**: 1k-sample subset, confirm denoise loss ↓ and `emb_std`
   stays ~1 (no collapse), rollout latents stay inside the tanh clamp.
3. **Short full run** on the tennis dataset, same budget as an existing LeWM
   baseline epoch count.
4. **Probes**: retrain the state head (`scripts/train/train_statehead.py`) on
   JEDI latents and rerun the rollout probe. Success criterion: `ball_y` R²
   in *rollout* (not teacher-forced) beats the LeWM baseline — that's the
   metric JEDI's stochastic sampling should move.

## Step 7 — Paper extras, after the baseline works

- **Random switching**: with prob 0.5 condition on sampled (denoised) latents
  instead of encoder latents during training — reduces rollout exposure bias
  but needs in-loop sampling; add once step 6 passes.
  **DONE (2026-07-09)**: `JEDI.self_rollout` + per-batch switch in
  `jedi_forward`, config `diffusion.self_cond_prob: 0.5`, logged as
  `fit/self_cond`. Sampled context frames are detached; the first H real
  frames stay graph-connected so switched batches still train the encoder.
- **Reward/termination head** `R_psi(z)`: only needed if we move from
  goal-MSE planning to reward-based planning/RL; the tennis reward is in the
  dataset, so this is a small CE/MSE head when we want it.
- SIGReg-weight ablation (0 vs. current).

## Data caveat

The 400k tennis dataset was collected with the `PLAYER_X_ADDR` swap bug
(heuristic never actually chased the ball). Fine for validating the JEDI
mechanics, but recollect before drawing dynamics-quality conclusions vs. LeWM.
