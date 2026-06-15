#!/usr/bin/env python3
"""
monitor.py — Live training monitor for Dronemania vision RL.

Runs alongside self_train.py (in a separate terminal).  Watches the vision
checkpoint for updates, loads the latest weights, and plays rendered episodes
using the full 3-D cockpit view from visualize_race.py — so you can see exactly
what the evolving policy is doing without interrupting training.

Usage
-----
    # In terminal A (training):
    python self_train.py --vision

    # In terminal B (monitor — this script):
    python monitor.py           # auto-loads rl_vision_policy_latest.zip
    python monitor.py --gps     # monitor the GPS model instead

Controls (OpenCV window)
    Q / ESC   quit
    R         force reload checkpoint now
    SPACE     pause / resume current episode
    N         skip to next episode immediately
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_simulator import RacingSimulator
from simulation.drone_gym_env   import (
    _KP_ROLL, _KD_ROLL, _KP_PITCH, _KD_PITCH, _KP_YAW, _KD_YAW,
    _MAX_TILT, _HOVER, _VX_MAX, _VY_MAX, _VZ_MAX, _YAW_MAX,
)
from perception.gate_detector import SimGateDetector

# Import cockpit draw functions from the existing visualizer
from visualize_race import (
    draw_cockpit_targeting,
    draw_detection_overlay,
    draw_gate_flash,
    draw_record_hud,
    draw_telemetry,
    load_records,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = pathlib.Path(__file__).parent
VISION_CKPT      = ROOT / "trained_models" / "rl_vision_policy_latest.zip"
GPS_CKPT         = ROOT / "trained_models" / "rl_policy_latest.zip"
VISION_TRAIN_LOG = ROOT / "records" / "vision_training_log.jsonl"
GPS_TRAIN_LOG    = ROOT / "records" / "training_log.jsonl"
RECORDS_FILE     = ROOT / "records" / "race_records.json"

# ── Spawn helper: mirrors RacingEnv.reset() so monitor matches training ──────────
def _spawn_at_gate0(sim: RacingSimulator) -> None:
    """Place drone 2 m in front of gate 0 with zero velocity.
    Must be called AFTER sim.reset() so current_gate_idx is 0."""
    g0 = sim.gates[0]
    spawn = np.array([g0.position[0] - 2.0, g0.position[1], g0.position[2]])
    sim.physics.set_state(
        position         = spawn,
        velocity         = np.zeros(3),
        orientation      = np.zeros(3),
        angular_velocity = np.zeros(3),
    )
    sim._prev_drone_pos = spawn.copy()


# ── PD inner loop (mirrors drone_gym_env._velocity_to_motors) ─────────────────
def velocity_to_motors(state: dict, vx: float, vy: float,
                        vz: float, yaw_rate: float,
                        prev_pitch: float, prev_roll: float):
    roll, pitch, yaw = state['orientation']
    ang_vel   = state['angular_velocity']
    drone_vel = state['velocity']

    tilt = np.sqrt(pitch**2 + roll**2)
    comp = 1.0 / max(0.87, np.cos(tilt))
    vz_err  = vz - drone_vel[2]
    alt_thr = np.clip((_HOVER + vz_err * 0.45 - drone_vel[2] * 0.50) * comp,
                      0.32, 0.72)

    vel_err = np.array([vx - drone_vel[0], vy - drone_vel[1]])
    cy, sy  = np.cos(yaw), np.sin(yaw)
    fwd_err =  vel_err[0]*cy + vel_err[1]*sy
    lat_err = -vel_err[0]*sy + vel_err[1]*cy

    max_r = np.radians(2.5)
    rp    = np.clip( fwd_err * 0.60, -_MAX_TILT,  _MAX_TILT)
    rr    = np.clip(-lat_err * 0.65, -_MAX_TILT,  _MAX_TILT)
    tp    = np.clip(rp, prev_pitch - max_r, prev_pitch + max_r)
    tr    = np.clip(rr, prev_roll  - max_r, prev_roll  + max_r)

    rc = _KP_ROLL  * (roll  - tr) + _KD_ROLL  * ang_vel[0]
    pc = _KP_PITCH * (pitch - tp) + _KD_PITCH * ang_vel[1]
    yc = np.clip(_KP_YAW * (yaw_rate - ang_vel[2]) - _KD_YAW * ang_vel[2],
                 -0.08, 0.08)

    motors = np.clip([alt_thr - pc + rc - yc,
                      alt_thr + pc + rc + yc,
                      alt_thr - pc - rc + yc,
                      alt_thr + pc - rc - yc], 0.2, 1.0)
    return motors, tp, tr


# ── Camera observation (mirrors drone_gym_env._get_camera_obs) ────────────────
def get_camera_obs(sim: RacingSimulator, detector: SimGateDetector) -> np.ndarray:
    state   = sim.physics.get_state()
    vel     = state['velocity']
    ori     = state['orientation']
    ang_vel = state['angular_velocity']
    total   = len(sim.gates)
    gi      = sim.current_gate_idx

    frame = sim.get_observation().get('camera')
    if frame is not None:
        det    = detector.detect(frame)
        cam_vec = det['obs_vector']
        dist_n  = float(det['dist_estimate']) / 60.0
    else:
        cam_vec = np.zeros(5, dtype=np.float32)
        dist_n  = 1.0
        det     = None

    obs = np.concatenate([
        cam_vec,
        np.clip(vel,     -20, 20),
        ori,
        np.clip(ang_vel, -15, 15),
        [np.clip(dist_n,  0,   1)],
        [gi / max(total, 1)],
        np.zeros(4, dtype=np.float32),
    ]).astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0), frame, det


# ── Training stats reader ─────────────────────────────────────────────────────
def read_latest_stats(log_path: pathlib.Path) -> dict:
    """Read the last line of the JSONL training log."""
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding='utf-8').strip().splitlines()
        if lines:
            return json.loads(lines[-1])
    except Exception:
        pass
    return {}


# ── Training stats HUD panel ──────────────────────────────────────────────────
def draw_training_panel(img: np.ndarray, stats: dict, ckpt_age_s: float,
                         model_label: str, new_model_flash: int) -> np.ndarray:
    """Overlay a compact training-stats panel on the right side."""
    h, w = img.shape[:2]
    pw, ph = 210, 165
    x0, y0 = w - pw - 5, h - ph - 5

    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    border_col = (0, 255, 160) if new_model_flash > 0 else (60, 60, 100)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), border_col, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lh   = 22

    title = "NEW MODEL LOADED" if new_model_flash > 0 else f"TRAINING  [{model_label}]"
    title_col = (0, 255, 160) if new_model_flash > 0 else (160, 200, 255)
    cv2.putText(img, title, (x0 + 6, y0 + 16), font, 0.38, title_col, 1)

    if stats:
        phase = stats.get('phase', '?')
        steps = stats.get('total_steps', 0)
        avg_g = stats.get('avg_gates', 0.0)
        best  = stats.get('best_clean_s')
        scrip = stats.get('scripted_best_s')
        trend = stats.get('trend', '')
        # Colour-code trend
        if '↑' in trend:
            trend_col = (0, 220, 80)
        elif '↓' in trend:
            trend_col = (0, 80, 255)
        else:
            trend_col = (180, 180, 180)

        rows = [
            (f"Phase  {phase}   Steps {steps//1000}k", (200, 200, 200)),
            (f"Avg gates   {avg_g:.1f} / 5",           (100, 220, 255)),
            (trend,                                     trend_col),
            (f"RL best  {best:.2f}s" if best else "RL best  --",
             (80, 255, 80) if best else (120, 120, 120)),
            (f"Scripted {scrip:.2f}s" if scrip else "Scripted best  --",
             (200, 200, 80)),
            (f"Ckpt age  {ckpt_age_s:.0f}s",           (120, 120, 160)),
        ]
        for i, (txt, col) in enumerate(rows):
            cv2.putText(img, txt, (x0 + 6, y0 + 16 + (i + 1) * lh),
                        font, 0.38, col, 1)
    else:
        cv2.putText(img, "Waiting for training log...", (x0 + 6, y0 + 40),
                    font, 0.35, (120, 120, 120), 1)
    return img


# ── Grid-view constants & helpers ─────────────────────────────────────────────
CELL_W, CELL_H = 320, 240   # native camera resolution per grid cell


def _cell_overlay(cell: np.ndarray, idx: int, gates: int, total: int,
                   gate_idx: int, speed: float, ep: int, done: bool) -> np.ndarray:
    """Minimal info overlay on a 320x240 grid cell."""
    border = (60, 60, 70) if done else (0, 180, 80)
    cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), border, 2)
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cell, f"#{idx+1}",         (4, 14),           fnt, 0.42, (180, 180, 180), 1)
    label = f"G{gate_idx+1}/{total}"
    tw = cv2.getTextSize(label, fnt, 0.42, 1)[0][0]
    cv2.putText(cell, label, (CELL_W // 2 - tw // 2, 14),     fnt, 0.42, (80, 210, 255),  1)
    cv2.putText(cell, f"ep{ep}",            (CELL_W - 36, 14), fnt, 0.33, (120, 120, 140), 1)
    cv2.putText(cell, f"{speed:.1f}m/s",   (4, CELL_H - 5),   fnt, 0.38, (200, 200, 80),  1)
    gcol = (80, 220, 80) if gates > 0 else (100, 100, 120)
    cv2.putText(cell, f"{gates}gts",       (CELL_W - 44, CELL_H - 5), fnt, 0.38, gcol, 1)
    if done:
        cv2.putText(cell, "DONE", (CELL_W // 2 - 22, CELL_H // 2),
                    fnt, 0.55, (60, 60, 180), 1)
    return cell


def _stats_bar(width: int, height: int, stats: dict, ckpt_age: float,
               model_label: str, new_model_flash: int, total_ep: int) -> np.ndarray:
    """Bottom stats strip for grid view."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(bar, (0, 0), (width, 0), (50, 50, 70), 1)
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    if new_model_flash > 0:
        cv2.putText(bar, "NEW MODEL LOADED", (8, height // 2 + 6),
                    fnt, 0.55, (0, 255, 160), 1)
    elif stats:
        phase = stats.get('phase', '?')
        steps = stats.get('total_steps', 0)
        avg_g = stats.get('avg_gates', 0.0)
        trend = stats.get('trend', '')
        tcol  = ((0, 220, 80)  if '\u2191' in trend else
                 (0, 80, 255)  if '\u2193' in trend else (160, 160, 160))
        line1 = (f"[{model_label}]  Phase {phase}  "
                 f"{steps//1000}k steps  Avg {avg_g:.1f}/5 gates  {trend}")
        line2 = (f"Ckpt age {ckpt_age:.0f}s  |  {total_ep} ep total  |  "
                 f"R=reload  SPC=pause  N=reset all  Q=quit")
        cv2.putText(bar, line1, (8, 18), fnt, 0.38, (180, 200, 220), 1)
        cv2.putText(bar, line2, (8, 36), fnt, 0.33, (100, 100, 130), 1)
    else:
        cv2.putText(bar, "Waiting for training log...  |  Q=quit",
                    (8, height // 2 + 5), fnt, 0.38, (100, 100, 120), 1)
    return bar


def _build_grid_display(frames_cache, n_grid, COLS, ROWS, cell_w, cell_h,
                         sims, gates_passed, episodes, is_done,
                         log_path, ckpt_mtime, model_label,
                         new_model_flash, total_ep, stats_h, win_w):
    cells = []
    for i in range(n_grid):
        cell = frames_cache[i].copy()
        st   = sims[i].physics.get_state()
        spd  = float(np.linalg.norm(st['velocity']))
        cell = _cell_overlay(cell, i, gates_passed[i], len(sims[i].gates),
                              sims[i].current_gate_idx, spd, episodes[i], is_done[i])
        cells.append(cell)
    while len(cells) < ROWS * COLS:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    grid = np.vstack(rows)
    stats    = read_latest_stats(log_path)
    ckpt_age = time.time() - ckpt_mtime if ckpt_mtime > 0 else 9999
    bar      = _stats_bar(win_w, stats_h, stats, ckpt_age, model_label,
                          new_model_flash, total_ep)
    return np.vstack([grid, bar])


# ── Checkpoint watcher (background thread) ────────────────────────────────────
class CheckpointWatcher(threading.Thread):
    def __init__(self, path: pathlib.Path, poll_secs: float = 5.0):
        super().__init__(daemon=True)
        self.path       = path
        self.poll_secs  = poll_secs
        self._mtime     = 0.0
        self.updated    = threading.Event()

    def run(self):
        while True:
            try:
                mt = self.path.stat().st_mtime if self.path.exists() else 0.0
                if mt > self._mtime + 1.0:
                    self._mtime = mt
                    self.updated.set()
            except Exception:
                pass
            time.sleep(self.poll_secs)

    @property
    def mtime(self):
        return self._mtime


# ── Grid monitor loop ─────────────────────────────────────────────────────────
def run_monitor_grid(vision: bool = True, n_grid: int = 4):
    """2x2 (or NxM) grid: N independent drone instances running simultaneously."""
    from stable_baselines3 import PPO
    COLS     = 2
    ROWS     = (n_grid + COLS - 1) // COLS
    STATS_H  = 50
    WIN_W    = COLS * CELL_W
    WIN_H    = ROWS * CELL_H + STATS_H
    SUBSTEPS = 4

    ckpt_path   = VISION_CKPT if vision else GPS_CKPT
    log_path    = VISION_TRAIN_LOG if vision else GPS_TRAIN_LOG
    model_label = "VISION" if vision else "GPS"

    print(f"[MONITOR] Grid {COLS}x{ROWS} ({n_grid} instances) | {ckpt_path.name}")
    print(f"[MONITOR] Q=quit  R=reload  SPACE=pause  N=reset all")

    sim_cfg = {'sim_rate': 60, 'perception': {'enabled': False},
               'planning': {'enabled': False}, 'fast_render': True}
    sims      = [RacingSimulator(dict(sim_cfg)) for _ in range(n_grid)]
    detectors = [SimGateDetector() for _ in range(n_grid)]

    def load_model():
        if not ckpt_path.exists():
            return None
        try:
            return PPO.load(str(ckpt_path.with_suffix('')))
        except Exception as e:
            print(f"[MONITOR] Load error: {e}")
            return None

    model = load_model()
    if model is None:
        print("[MONITOR] Waiting for checkpoint to appear...")
        while not ckpt_path.exists():
            time.sleep(2.0)
        model = load_model()

    watcher = CheckpointWatcher(ckpt_path, poll_secs=4.0)
    watcher.start()

    cv2.namedWindow('Dronemania Monitor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Dronemania Monitor', WIN_W, WIN_H)

    prev_pitches  = [0.0]  * n_grid
    prev_rolls    = [0.0]  * n_grid
    gates_passed  = [0]    * n_grid
    episodes      = [0]    * n_grid
    is_done       = [True] * n_grid   # True → triggers reset on next tick
    frames_cache  = [np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)] * n_grid

    new_model_flash = 0
    total_ep        = 0
    paused          = False

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                nm = load_model()
                if nm:
                    model = nm
                    new_model_flash = 90
                    print("[MONITOR] Manual reload")
            elif key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                is_done = [True] * n_grid

            if watcher.updated.is_set() or model is None:
                watcher.updated.clear()
                nm = load_model()
                if nm is not None:
                    model = nm
                    new_model_flash = 90
                    import datetime as _dt
                    _ts = _dt.datetime.now().strftime('%H:%M:%S')
                    print(f"[MONITOR\u2190TRAINER] {_ts} | New checkpoint -> {ckpt_path.name}")

            if model is None or paused:
                disp = _build_grid_display(
                    frames_cache, n_grid, COLS, ROWS, CELL_W, CELL_H,
                    sims, gates_passed, episodes, is_done,
                    log_path, watcher.mtime, model_label,
                    new_model_flash, total_ep, STATS_H, WIN_W)
                cv2.imshow('Dronemania Monitor', disp)
                time.sleep(0.033)
                continue

            for i in range(n_grid):
                if is_done[i]:
                    sims[i].reset()
                    _spawn_at_gate0(sims[i])
                    gates_passed[i] = 0
                    prev_pitches[i] = prev_rolls[i] = 0.0
                    is_done[i]      = False
                    episodes[i]    += 1
                    total_ep       += 1

                obs, frm, _ = get_camera_obs(sims[i], detectors[i])
                if frm is not None:
                    frames_cache[i] = frm.copy()

                action, _ = model.predict(obs, deterministic=True)
                vx  = float(action[0]) * _VX_MAX
                vy  = float(action[1]) * _VY_MAX
                vz  = float(action[2]) * _VZ_MAX
                yaw = float(action[3]) * _YAW_MAX

                gate_info = {}
                for _ in range(SUBSTEPS):
                    st = sims[i].physics.get_state()
                    motors, prev_pitches[i], prev_rolls[i] = velocity_to_motors(
                        st, vx, vy, vz, yaw, prev_pitches[i], prev_rolls[i])
                    sims[i].set_motor_commands(motors)
                    gi = sims[i].check_gates()
                    if gi:
                        gate_info = gi
                    t, tr, _ = sims[i].check_termination()
                    if t or tr:
                        is_done[i] = True
                        break

                if 'gate_passed' in gate_info:
                    gates_passed[i] += 1

            disp = _build_grid_display(
                frames_cache, n_grid, COLS, ROWS, CELL_W, CELL_H,
                sims, gates_passed, episodes, is_done,
                log_path, watcher.mtime, model_label,
                new_model_flash, total_ep, STATS_H, WIN_W)
            cv2.imshow('Dronemania Monitor', disp)
            if new_model_flash > 0:
                new_model_flash -= 1

    except KeyboardInterrupt:
        print("\n[MONITOR] Quit")
    finally:
        cv2.destroyAllWindows()


# ── Single-instance monitor (original cockpit view) ───────────────────────────
def run_monitor_single(vision: bool = True):
    from stable_baselines3 import PPO

    ckpt_path = VISION_CKPT if vision else GPS_CKPT
    log_path  = VISION_TRAIN_LOG if vision else GPS_TRAIN_LOG
    model_label = "VISION" if vision else "GPS"

    print(f"[MONITOR] Watching: {ckpt_path.name}")
    print(f"[MONITOR] Press Q/ESC=quit  R=reload  SPACE=pause  N=next episode")

    # ── Build sim (full render — not fast_render) ─────────────────────────────
    sim_cfg = {'sim_rate': 60, 'perception': {'enabled': False},
               'planning': {'enabled': False}, 'fast_render': False}
    sim      = RacingSimulator(sim_cfg)
    detector = SimGateDetector()

    # ── Load model ────────────────────────────────────────────────────────────
    def load_model():
        if not ckpt_path.exists():
            return None
        try:
            from stable_baselines3 import PPO
            return PPO.load(str(ckpt_path.with_suffix('')))
        except Exception as e:
            print(f"[MONITOR] Could not load model: {e}")
            return None

    model = load_model()
    if model is None:
        print(f"[MONITOR] No checkpoint found at {ckpt_path}")
        print(f"          Start training first:  python self_train.py --vision")
        print(f"          Waiting for checkpoint to appear …")
        while not ckpt_path.exists():
            time.sleep(2.0)
        model = load_model()

    # ── Watcher thread ────────────────────────────────────────────────────────
    watcher     = CheckpointWatcher(ckpt_path, poll_secs=4.0)
    watcher.start()

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow('Dronemania Monitor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Dronemania Monitor', 960, 720)

    records          = load_records()
    gate_splits: list = []
    new_model_flash   = 0   # countdown frames for "NEW MODEL" banner

    episode   = 0
    try:
        while True:
            # ── Check for checkpoint update ───────────────────────────────────
            if watcher.updated.is_set() or model is None:
                watcher.updated.clear()
                new_m = load_model()
                if new_m is not None:
                    model = new_m
                    new_model_flash = 90
                    import datetime as _dt
                    _ts = _dt.datetime.now().strftime('%H:%M:%S')
                    print(f"[MONITOR←TRAINER] {_ts} | New checkpoint detected → loaded {ckpt_path.name}")

            if model is None:
                time.sleep(1.0)
                continue

            # ── New episode ───────────────────────────────────────────────────
            episode += 1
            sim.reset()
            _spawn_at_gate0(sim)
            gate_splits = []
            gate_flash  = {'timer': 0, 'color': (0, 255, 0), 'text': '', 'subtext': ''}
            prev_pitch = prev_roll = 0.0
            gates_passed    = 0
            race_start      = time.time()
            paused          = False
            skip_episode    = False
            records         = load_records()
            print(f"\n[MONITOR] Episode {episode}  (model: {ckpt_path.name})")

            SUBSTEPS = 4   # match training

            while True:
                # ── Key handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    raise KeyboardInterrupt
                elif key == ord('r'):
                    new_m = load_model()
                    if new_m:
                        model = new_m
                        new_model_flash = 90
                        print("[MONITOR] Manual reload")
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('n'):
                    skip_episode = True

                if skip_episode:
                    break

                if paused:
                    # Still render + show PAUSED
                    frame = sim.get_observation().get('camera',
                                np.zeros((240, 320, 3), dtype=np.uint8))
                    vis = _make_vis(sim, frame, None, None, drone_state_cache,
                                    gate_flash, gates_passed, gate_splits,
                                    records, race_start, paused,
                                    log_path, watcher.mtime, model_label,
                                    new_model_flash)
                    cv2.imshow('Dronemania Monitor', vis)
                    time.sleep(0.033)
                    continue

                # ── RL step (SUBSTEPS physics ticks per action) ───────────────
                obs, frame, det_result = get_camera_obs(sim, detector)
                action, _ = model.predict(obs, deterministic=True)

                vx  = float(action[0]) * _VX_MAX
                vy  = float(action[1]) * _VY_MAX
                vz  = float(action[2]) * _VZ_MAX
                yaw = float(action[3]) * _YAW_MAX

                gate_info = {}
                for _ in range(SUBSTEPS):
                    state  = sim.physics.get_state()
                    motors, prev_pitch, prev_roll = velocity_to_motors(
                        state, vx, vy, vz, yaw, prev_pitch, prev_roll)
                    sim.set_motor_commands(motors)
                    gi = sim.check_gates()
                    if gi:
                        gate_info = gi
                    terminated, truncated, term_info = sim.check_termination()
                    if terminated or truncated:
                        break

                drone_state_cache = sim.physics.get_state()

                # Gate flash
                if 'gate_passed' in gate_info:
                    gates_passed += 1
                    lat  = gate_info.get('gate_lateral_offset', 0.0)
                    vert = gate_info.get('gate_vertical_offset', 0.0)
                    gate_splits.append(time.time() - race_start)
                    gate_flash['timer']   = 70
                    gate_flash['color']   = (0, 220, 0)
                    gate_flash['text']    = f"GATE {gates_passed} CLEAR"
                    gate_flash['subtext'] = f"lat:{lat:+.2f}m  vert:{vert:+.2f}m"
                    print(f"  Gate {gates_passed}/5 cleared")
                elif 'gate_missed' in gate_info:
                    g  = gate_info['gate_missed']
                    gate_flash['timer']   = 100
                    gate_flash['color']   = (0, 60, 255)
                    gate_flash['text']    = f"GATE {g+1} MISSED"
                    gate_flash['subtext'] = ""

                drone_state_cache = {'position':         sim.physics.get_state()['position'],
                                     'velocity':         sim.physics.get_state()['velocity'],
                                     'orientation':      sim.physics.get_state()['orientation'],
                                     'angular_velocity': sim.physics.get_state()['angular_velocity']}

                # ── Render ────────────────────────────────────────────────────
                vis = _make_vis(sim, frame, det_result, obs,
                                drone_state_cache,
                                gate_flash, gates_passed, gate_splits,
                                records, race_start, paused,
                                log_path, watcher.mtime, model_label,
                                new_model_flash)
                cv2.imshow('Dronemania Monitor', vis)
                if new_model_flash > 0:
                    new_model_flash -= 1

                if terminated or truncated:
                    elapsed = time.time() - race_start
                    is_clean = gates_passed == len(sim.gates)
                    result_txt = (f"COMPLETE {elapsed:.2f}s" if is_clean
                                  else f"ENDED: {term_info.get('reason','?').upper()}")
                    color = (0, 220, 0) if is_clean else (0, 60, 255)
                    cv2.putText(vis, result_txt,
                                (vis.shape[1]//2 - 260, vis.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.imshow('Dronemania Monitor', vis)
                    print(f"  [RESULT] {result_txt}  gates:{gates_passed}/5")
                    cv2.waitKey(2500)
                    break

    except KeyboardInterrupt:
        print("\n[MONITOR] Quit")
    finally:
        cv2.destroyAllWindows()


def _make_vis(sim, frame, det_result, obs,
              drone_state, gate_flash, gates_passed, gate_splits,
              records, race_start, paused,
              log_path, ckpt_mtime, model_label, new_model_flash):
    """Compose the full cockpit frame, same pipeline as visualize_race.py."""

    # ── Build drone_state dict for draw functions ─────────────────────────────
    ds = drone_state if drone_state else {
        'position': np.zeros(3), 'velocity': np.zeros(3),
        'orientation': np.zeros(3), 'angular_velocity': np.zeros(3)
    }

    # ── Build perception dict for draw_detection_overlay ─────────────────────
    if det_result is not None and det_result['gate_visible']:
        cx, cy, w, h = det_result['cx'], det_result['cy'], det_result['w'], det_result['h']
        ng = {'center': (cx, cy), 'bbox': (cx - w//2, cy - h//2, w, h),
              'area': det_result['area'], 'confidence': min(1.0, det_result['area']/4000),
              'dist_estimate': det_result['dist_estimate']}
        perception = {'gates': [ng], 'next_gate': ng,
                      'num_detected': 1, 'confidence': ng['confidence']}
    else:
        perception = {'gates': [], 'next_gate': None, 'num_detected': 0, 'confidence': 0.0}

    # target dict for speed-zone badge
    gi = sim.current_gate_idx
    if gi < len(sim.gates):
        gate_pos = sim.gates[gi].position
        gate_dist = float(np.linalg.norm(gate_pos - ds['position']))
    else:
        gate_pos  = ds['position']
        gate_dist = 0.0
    target = {'position': gate_pos, 'gate_distance': gate_dist}

    # ── Start with camera frame ───────────────────────────────────────────────
    vis = (frame.copy() if frame is not None
           else np.zeros((240, 320, 3), dtype=np.uint8))

    # Upscale 320×240 → 960×720
    vis = cv2.resize(vis, (960, 720), interpolation=cv2.INTER_NEAREST)

    # overlays (same order as visualize_race.py)
    vis = draw_detection_overlay(vis, perception, ds, target, scale=3)

    target_gate = sim.gates[gi] if gi < len(sim.gates) else None
    vis = draw_cockpit_targeting(vis, ds, target_gate, gi, len(sim.gates),
                                 fov_degrees=sim.fov_degrees)
    vis = draw_gate_flash(vis, gate_flash)

    elapsed = time.time() - race_start
    vis = draw_record_hud(vis, elapsed, gates_passed, len(sim.gates),
                          records, gate_splits)
    vis = draw_telemetry(vis, ds, target, gates_passed, len(sim.gates), elapsed)

    # ── Training stats panel ──────────────────────────────────────────────────
    stats     = read_latest_stats(log_path)
    ckpt_age  = time.time() - ckpt_mtime if ckpt_mtime > 0 else 9999
    vis = draw_training_panel(vis, stats, ckpt_age, model_label, new_model_flash)

    # ── RL policy badge (top-centre) ──────────────────────────────────────────
    badge = f"RL POLICY [{model_label}]  Gate {gi+1}/{len(sim.gates)}"
    bsz   = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    bx    = vis.shape[1]//2 - bsz[0]//2
    cv2.putText(vis, badge, (bx, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)
    cv2.putText(vis, "R=reload  SPACE=pause  N=next  Q=quit",
                (bx, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 160), 1)

    if paused:
        cv2.putText(vis, "PAUSED",
                    (vis.shape[1]//2 - 80, vis.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    return vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dronemania live training monitor")
    parser.add_argument("--gps",    action="store_true",
                        help="Monitor GPS policy instead of vision policy")
    parser.add_argument("--single", action="store_true",
                        help="Single-instance cockpit view (default: 2x2 grid)")
    parser.add_argument("--grid",   type=int, default=4, metavar="N",
                        help="Number of grid instances (default 4, ignored with --single)")
    args = parser.parse_args()
    if args.single:
        run_monitor_single(vision=not args.gps)
    else:
        run_monitor_grid(vision=not args.gps, n_grid=args.grid)
