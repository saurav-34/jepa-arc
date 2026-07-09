"""Realtime 1v1 Tennis in the browser — human (keyboard, bottom) vs JEPA (top).

Headless-server version of tennis_human_vs_jepa.py: instead of a pygame
window, the game loop runs server-side and streams JPEG frames over a
WebSocket to a canvas in your browser; keyboard state is sent back the
other way. Same referee (real PettingZoo tennis_v3), same JEPA pipeline
(frozen LeWM encoder -> linear state head -> reactive intercept policy).

Usage (on the server):
    python scripts/play/tennis_human_vs_jepa_web.py \
        [--wm PATH] [--statehead DIR] [--fps 15] [--port 8765]

Then from your local machine:
    ssh -L 8765:localhost:8765 <user>@<server>
    open http://localhost:8765

Controls (in the browser tab):  ←/→ move   SPACE swing/serve
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from aiohttp import WSMsgType, web

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--wm', type=str, default=None,
                    help='World-model weights (.pt). Default: highest-epoch file in '
                         '$STABLEWM_HOME/checkpoints/lewm_tennis/.')
parser.add_argument('--statehead', type=str, default=None,
                    help='Statehead folder. Default: $STABLEWM_HOME/checkpoints/statehead_tennis/.')
parser.add_argument('--fps', type=int, default=15,
                    help='Control steps per second. 15 * frameskip 4 = 60 ALE fps (real time).')
parser.add_argument('--frameskip', type=int, default=4,
                    help='ALE frames per control step. MUST match the data collection stride (4).')
parser.add_argument('--deadzone', type=float, default=6.0,
                    help='|ball_x - player_x| (RAM px) within which JEPA swings instead of moving.')
parser.add_argument('--host', type=str, default='127.0.0.1',
                    help='Bind address. Keep 127.0.0.1 and reach it via `ssh -L`.')
parser.add_argument('--port', type=int, default=8765)
parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

IMG_SIZE  = 128
EMBED_DIM = 192

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ALE full-action-set ids (same as collector)
NOOP, FIRE, RIGHT, LEFT = 0, 1, 3, 4
# env agent ids. Roles are anchored to court ENDS, not agents: the ROM swaps
# the agents' ends after odd games (real tennis rules), so which agent the
# keyboard/JEPA drives flips with it — human is always the BOTTOM paddle,
# JEPA always the TOP.
FIRST_0, SECOND_0 = 'first_0', 'second_0'

# ──────────────────────────────────────────────────────────────
# Loading (same as tennis_human_vs_jepa.py)
# ──────────────────────────────────────────────────────────────

def _stablewm_home() -> Path:
    return Path(os.path.expanduser(os.getenv('STABLEWM_HOME', '~/.stable_worldmodel')))


def _latest_pt(folder: Path) -> Path:
    pts = list(folder.glob('weights_epoch_*.pt'))
    if not pts:
        raise FileNotFoundError(f'no weights_epoch_*.pt in {folder}')
    return max(pts, key=lambda p: int(p.stem.rsplit('_', 1)[-1]))


def load_models():
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from stable_worldmodel.wm.utils import load_pretrained

    wm_path = Path(args.wm) if args.wm else _latest_pt(_stablewm_home() / 'checkpoints' / 'lewm_tennis')
    print(f'World model: {wm_path}')
    model = load_pretrained(str(wm_path)).to(args.device).eval()

    head_dir = Path(args.statehead) if args.statehead else _stablewm_home() / 'checkpoints' / 'statehead_tennis'
    with open(head_dir / 'state_cols.json') as f:
        cols = json.load(f)
    head = nn.Linear(EMBED_DIM, len(cols))
    head.load_state_dict(torch.load(head_dir / 'statehead.pt', map_location='cpu'))
    head = head.to(args.device).eval()
    mean = np.load(head_dir / 'state_mean.npy')
    std  = np.load(head_dir / 'state_std.npy')
    print(f'State head: {head_dir}  cols={cols}')
    return model, head, cols, mean, std


# ──────────────────────────────────────────────────────────────
# JEPA side: pixels -> belief -> action
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_belief(model, head, cols, mean, std, frame_rgb: np.ndarray) -> dict:
    """Raw ALE frame (210x160x3 uint8) -> decoded state dict (RAM units)."""
    x = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    t = torch.from_numpy(x).unsqueeze(0).to(args.device,
                                            dtype=next(model.encoder.parameters()).dtype)
    cls = model.encoder(t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    z   = model.projector(cls).float()
    raw = head(z).squeeze(0).cpu().numpy() * std + mean
    return dict(zip(cols, raw.tolist()))


def jepa_policy(belief: dict, deadzone: float, own_col: str) -> int:
    """Reactive intercept: chase decoded ball_x with own paddle, swing when aligned.

    `own_col` is the decoded column of the paddle JEPA controls — always
    'enemy_x' (RAM[27], top court), since roles are anchored to court ends.
    """
    dx = belief['ball_x'] - belief[own_col]
    if abs(dx) <= deadzone:
        return FIRE
    return RIGHT if dx > 0 else LEFT


def games_total(ale) -> int:
    """Cumulative games won by both agents (RAM[71] + RAM[72])."""
    r = ale.getRAM()
    return int(r[71]) + int(r[72])


def first0_is_bottom(ale) -> bool:
    """Current end assignment. Agents change ends after odd-numbered cumulative
    games (verified by wiggle-probe: game 1 first_0=RAM[26]=bottom, games 2-3
    top, games 4-5 bottom, game 6 top, ...)."""
    return ((games_total(ale) + 1) // 2) % 2 == 0


# ──────────────────────────────────────────────────────────────
# Browser page
# ──────────────────────────────────────────────────────────────

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tennis — you vs JEPA</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#ddd;
            font-family:monospace;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:8px}
  #wrap{position:relative;height:80vh;aspect-ratio:160/210}
  canvas{image-rendering:pixelated;width:100%;height:100%;
         background:#000;outline:2px solid #333}
  #you{position:absolute;right:100%;padding-right:10px;font-size:16px;
       font-weight:bold;color:#6f6;transition:top .6s;white-space:nowrap}
  #side{font-size:18px;font-weight:bold;color:#6f6}
  #side.flash{animation:pulse 1.2s ease-in-out 3}
  @keyframes pulse{50%{color:#ff5;transform:scale(1.15)}}
  #hud{font-size:14px;white-space:pre}
  #status{font-size:13px;color:#888}
</style></head><body>
<div id="side">connecting…</div>
<div id="wrap">
  <div id="you" style="top:75%">YOU ▶</div>
  <canvas id="c" width="160" height="210"></canvas>
</div>
<div id="hud"></div>
<div id="status">←/→ move &nbsp; SPACE swing/serve &nbsp; you always play the BOTTOM paddle</div>
<script>
const ctx = document.getElementById('c').getContext('2d');
const hud = document.getElementById('hud');
const side = document.getElementById('side');
const you = document.getElementById('you');
let curSide = null;
const keys = {left:false, right:false, space:false};
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.binaryType = 'blob';

ws.onmessage = async (ev) => {
  if (typeof ev.data === 'string') {
    const m = JSON.parse(ev.data);
    hud.textContent = m.hud;
    if (m.you !== curSide) {
      curSide = m.you;
      side.textContent = `YOU ARE THE ${m.you.toUpperCase()} PLAYER`;
      you.style.top = m.you === 'bottom' ? '75%' : '15%';
      side.classList.remove('flash'); void side.offsetWidth; side.classList.add('flash');
    }
    return;
  }
  const bmp = await createImageBitmap(ev.data);
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
};
ws.onclose = () => { hud.textContent = 'disconnected — restart the server script and reload'; };

function send() { if (ws.readyState === 1) ws.send(JSON.stringify(keys)); }
const map = {ArrowLeft:'left', ArrowRight:'right', ' ':'space'};
addEventListener('keydown', e => {
  if (e.key in map) { e.preventDefault(); if (!keys[map[e.key]]) { keys[map[e.key]] = true; send(); } }
});
addEventListener('keyup', e => {
  if (e.key in map) { e.preventDefault(); keys[map[e.key]] = false; send(); }
});
addEventListener('blur', () => { for (const k in keys) keys[k] = false; send(); });
</script></body></html>
"""

# ──────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────

class Game:
    def __init__(self):
        self.keys = {'left': False, 'right': False, 'space': False}
        self.clients: set[web.WebSocketResponse] = set()

    def human_action(self) -> int:
        if self.keys['space']:
            return FIRE
        if self.keys['right']:
            return RIGHT
        if self.keys['left']:
            return LEFT
        return NOOP


async def game_loop(game: Game, model, head, cols, mean, std):
    from pettingzoo.atari import tennis_v3

    def _rom_dir():
        import ale_py
        return os.path.join(os.path.dirname(ale_py.__file__), 'roms')

    env = tennis_v3.parallel_env(render_mode='rgb_array', auto_rom_install_path=_rom_dir())
    env.reset(seed=int(time.time()) % 100000)
    ale = env.unwrapped.ale                      # referee info only (side assignment)
    frame = env.render()
    score = {'human': 0, 'jepa': 0}              # points won, by role
    games = {'human': 0, 'jepa': 0}              # games won, by role
    dt = 1.0 / args.fps

    print(f'\nMatch on! Open http://localhost:{args.port} (via ssh -L {args.port}:localhost:{args.port}).')

    while True:
        t_start = time.perf_counter()

        if not game.clients:                     # nobody watching -> pause the match
            await asyncio.sleep(0.2)
            continue

        # ── anchor roles to court ends: human=BOTTOM, JEPA=TOP, always ──
        # (the ROM swaps the agents' ends after odd games; we swap which agent
        # the keyboard/JEPA drive so each player keeps their side of the court)
        human_agent = FIRST_0 if first0_is_bottom(ale) else SECOND_0
        jepa_agent  = SECOND_0 if human_agent == FIRST_0 else FIRST_0
        jepa_col = 'enemy_x'                     # top-court paddle column

        # ── JEPA sees pixels, decodes belief, reacts ──
        t0 = time.perf_counter()
        belief = decode_belief(model, head, cols, mean, std, frame)
        a_jepa = jepa_policy(belief, args.deadzone, jepa_col)
        t_infer = (time.perf_counter() - t0) * 1000

        a_human = game.human_action()

        # ── advance the real game: hold both actions for `frameskip` ALE frames ──
        games_before = games_total(ale)
        for _ in range(args.frameskip):
            if not env.agents:
                break
            _, rewards, _, _, _ = env.step({jepa_agent: a_jepa, human_agent: a_human})
            for ag, r in rewards.items():
                if r > 0:
                    role = 'human' if ag == human_agent else 'jepa'
                    score[role] += 1
                    if games_total(ale) > games_before:   # this point closed a game
                        games[role] += 1
                        games_before = games_total(ale)
        if not env.agents:                       # match over -> new match
            env.reset()
        frame = env.render()

        # ── broadcast frame + HUD ──
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        act_name = {NOOP: '·', FIRE: 'SWING', RIGHT: '→', LEFT: '←'}.get(a_jepa, '?')
        hud = (f'points You {score["human"]} : {score["jepa"]} JEPA   '
               f'games You {games["human"]}-{games["jepa"]} JEPA   '
               f'belief ball_x={belief.get("ball_x", 0):5.1f} '
               f'jepa_x={belief.get(jepa_col, 0):5.1f} act={act_name:5s} '
               f'{t_infer:4.1f}ms')
        msg = json.dumps({'hud': hud, 'you': 'bottom'})
        for ws in list(game.clients):
            try:
                if ok:
                    await ws.send_bytes(jpg.tobytes())
                await ws.send_str(msg)
            except (ConnectionError, RuntimeError):
                game.clients.discard(ws)

        await asyncio.sleep(max(0.0, dt - (time.perf_counter() - t_start)))


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    game: Game = request.app['game']
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    game.clients.add(ws)
    print(f'client connected ({len(game.clients)} total)')
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    game.keys.update(json.loads(msg.data))
                except (json.JSONDecodeError, TypeError):
                    pass
    finally:
        game.clients.discard(ws)
        game.keys = {'left': False, 'right': False, 'space': False}
        print(f'client disconnected ({len(game.clients)} total)')
    return ws


async def index(_request: web.Request) -> web.Response:
    return web.Response(text=PAGE, content_type='text/html')


async def main():
    model, head, cols, mean, std = load_models()

    game = Game()
    app = web.Application()
    app['game'] = game
    app.add_routes([web.get('/', index), web.get('/ws', ws_handler)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print(f'Serving on http://{args.host}:{args.port}')

    try:
        await game_loop(game, model, head, cols, mean, std)
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nbye')
