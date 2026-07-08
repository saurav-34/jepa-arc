# Tennis LeWM — Evaluation Results

World model: `LeWM` (ViT-tiny encoder @128px/patch-8, embed_dim 192), trained on
`tennis_ma_128x128_400k.lance` (400k transitions, **random-policy** both agents).
Checkpoint probed: `weights_epoch_3.pt`. All evals use `CUDA_VISIBLE_DEVICES=1`,
`STABLEWM_HOME=.../jepa-arc/.stablewm_home`, 60k–120k frames.

Scripts:
- `scripts/eval/probe_tennis_state.py` — linear state probe (encoder & predictor space)
- `scripts/eval/horizon_tennis.py` — horizon sweep + action controllability
- `scripts/eval/rollout_accuracy_tennis.py` — autoregressive rollout accuracy vs persistence

---

## TL;DR

- The model is **not collapsed / not a shortcut**: a linear head reads ball & paddle
  positions from the embeddings to a few pixels. SIGReg dropping fast in ~2 epochs was
  normal, not a failure.
- **One step ahead the model is accurate (~2–3px).** The **paddle is controllable**
  (actions steer predicted `player_x` in the correct direction).
- **Autoregressive rollout drifts fast** (ball_x ~3px @h1 → ~10px @h4) and, for the
  slow-moving x-variables, is **worse than persistence** — because random-policy data
  has a near-static ball, so little real dynamics were learned.
- **Verdict:** long-horizon MPC interception = not yet. Short-horizon / reactive control
  (per-frame state estimation + move toward `ball_x`, re-anchor every step) = feasible now.
- **Root fixes for real MPC:** (1) recollect data with a competent policy (ball in play),
  (2) multi-step rollout training (feed predictions back / scheduled sampling).

---

## 1. Linear state probe — R² (val) per variable

`probe_tennis_state.py`, 120k frames. "encoder" = `projector(encoder(frame))`,
"predictor" = `pred_proj(predictor(ctx))[:, -1]` (the space the planner uses).

| variable  | encoder R² | predictor R² | MAE (px) | note |
|-----------|-----------:|-------------:|---------:|------|
| player_x  | 0.90 | 0.90 | ~3 | **controllable paddle** — well encoded |
| ball_x    | 0.67 | 0.67 | ~3.8 | weak link (planner aim point) |
| ball_y    | 0.91 | 0.66 | ~1–3.9 | softens encoder→predictor |
| ball_vy   | 0.40 | 0.83 | ~1.5 | predictor space captures vertical velocity |
| enemy_x   | 0.74 | 0.73 | ~3.4 | opponent paddle |
| player_y  | 0.08 | 0.04 | 0.53 | low-variance (paddle fixed in y) → R² is noise |
| enemy_y   | 0.15 | 0.10 | 0.56 | low-variance → noise |
| ball_vx   | 0.06 | 0.05 | 0.16 | low-variance → noise |

**Conclusion:** positions genuinely encoded. Low-R² vars are all low-variance (ignore).

## 2. Data geometry (variance of ground-truth state, 120k frames)

| var | mean | std | range |
|-----|-----:|----:|-------|
| ball_x | 71.7 | 18.5 | 2–150 |
| ball_y | 17.7 | 8.3 | 0–39 |
| ball_vx | -0.1 | 1.3 | -50–6 |
| ball_vy | 0.0 | 5.4 | -25–20 |
| player_x | 132.6 | **26.3** | 9–142 |
| player_y | 141.9 | 1.9 | 122–148 |
| enemy_x | 22.6 | 16.9 | 17–142 |
| enemy_y | 7.2 | 2.1 | 2–35 |

**Paddles move in x, are ~fixed in y → effectively horizontal-Pong.** Ball travels
mainly in y (vy std 5.4 ≫ vx std 1.3), i.e. nearly vertical crossing.

## 3. Controllability — which paddle does the fed `action` control?

Ground-truth Δposition grouped by action family (RIGHT={3,6,8,11,14,16}, LEFT={4,7,9,12,15,17}):

| action fed | Δplayer_x | Δenemy_x |
|------------|----------:|---------:|
| RIGHT-fam (`action`, second_0) | +0.65 | +0.02 |
| LEFT-fam  (`action`, second_0) | **-2.63** | -0.06 |
| RIGHT-fam (`action_human`, first_0) | +0.09 | +3.67 |
| LEFT-fam  (`action_human`, first_0) | -0.00 | -0.60 |

→ The model-fed `action` (second_0) controls **`player_x`** (the R²=0.90 paddle).
`action_human` (unfed) controls `enemy_x`. (The config comment mislabels these, but
functionally the JEPA agent controls the well-decoded paddle.)

**Model-side controllability** (`horizon_tennis.py`, swap last action, decode player_x):

| horizon | RIGHT | NOOP | LEFT | RIGHT−LEFT |
|---------|------:|-----:|-----:|-----------:|
| h=1 | 135.4 | 134.8 | 132.6 | **+2.8 px/step** |
| h=3 | 135.1 | 135.0 | 134.3 | +0.8 px/step |

→ Monotonic and correctly signed; magnitude matches real control (~2–3px/step).
**The model learned that the action steers the paddle.**

## 4. Horizon sweep — R² of predictor_emb → state at (last_ctx + h)

`horizon_tennis.py`, shuffled ridge split, 60k frames.

| h | player_x | ball_x | ball_y |
|---|---------:|-------:|-------:|
| 1 | 0.934 | 0.696 | 0.906 |
| 2 | 0.932 | 0.697 | 0.917 |
| 3 | 0.930 | 0.681 | 0.912 |
| 4 | 0.928 | 0.656 | 0.905 |
| 6 | 0.924 | 0.659 | 0.894 |

→ Flat (variables are autocorrelated); no clean horizon peak. Near-future is read well.
(An earlier apparent "peak at h=3" was an artifact of a sequential train/test split.)

## 5. Autoregressive rollout accuracy (the real planning path)

`rollout_accuracy_tennis.py`, 60k frames, 4000 starts, real actions fed, decode head
fit in **matched predictor space** (head R²: player_x 0.93, ball_x 0.73, ball_y 0.91).
MAE in pixels, `rollout | persistence`:

| h | player_x | ball_x | ball_y |
|---|---------:|-------:|-------:|
| 1 | 2.56 \| 0.98 | **3.02 \| 0.18** | 1.43 \| 4.57 |
| 2 | 6.00 \| 1.16 | 7.06 \| 0.45 | 3.55 \| 8.21 |
| 3 | 7.99 \| 1.33 | 7.46 \| 0.67 | 6.83 \| 10.88 |
| 4 | 9.17 \| 1.42 | 10.03 \| 0.90 | 7.33 \| 12.62 |
| 6 | 10.24 \| 1.64 | 11.01 \| 1.27 | 7.55 \| 12.99 |
| 8 | 11.26 \| 1.81 | 11.91 \| 1.54 | 7.55 \| 9.84 |
| 12 | 12.29 \| 2.06 | 12.93 \| 2.18 | 6.87 \| 6.74 |

Observations:
- **h=1 accurate (~2–3px).** Confirms one-step prediction is good.
- **Rollout compounds** — error roughly triples h1→h2, keeps climbing (OOD feedback:
  predictor consumes its own `pred_proj` outputs, which it was never trained on).
- **player_x / ball_x: rollout worse than persistence at all horizons** (those vars
  barely move in random data; rollout invents ~10px of spurious motion).
- **ball_y: rollout beats persistence at h=2–8** — the one dynamic it genuinely captured.

## 6. Verdict & next steps

**Can the JEPA side intercept the ball?**
- Long-horizon MPC (roll out 5–8 steps, score candidates): **no** — by the ~7–8 step
  lead time needed to reach a crossing ball, `ball_x` error is ~11px and rollouts lose
  to persistence.
- Short-horizon / reactive control: **yes, plausibly** — use the model as a per-frame
  state estimator (encode real frame → decode `ball_x`, `player_x`), move paddle toward
  `ball_x`, re-anchor every step (K=1), ≤2-step planning. Ball is near-vertical so
  tracking current `ball_x` suffices.

**Root fixes for real anticipatory MPC:**
1. Recollect data with a competent/scripted policy (ball actually in play → real dynamics).
2. Multi-step rollout training (feed predictions back during training / scheduled
   sampling) so the predictor is robust to consuming its own outputs.

**Environment fixes needed to run any of this** (see memory): `STABLEWM_HOME` to a
writable dir; PyTorch `+cu128` (driver 570 = CUDA 12.8, not `+cu130`); `trainer.devices=1`
(SIGReg is single-GPU); eval on `CUDA_VISIBLE_DEVICES=1`.
