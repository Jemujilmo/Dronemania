#!/usr/bin/env python3
"""
train.py — Clean PPO training loop (hard fork of self_train.py).

What changed vs self_train.py
==============================
  REMOVED:
    - Trainer lock / duplicate-instance guard  (_TRAINER_LOCK, _acquire_lock,
      _release_lock, _pid_alive).  The lock was causing persistent crash-loops
      whenever a prior run left a stale .lock or .corrupt file on Windows.
    - train_watchdog.ps1 / monitor_watchdog.ps1 integration.  The watchdog
      scripts were silently spawning competing instances that clobbered each
      other's checkpoints.  This file is designed to be run once, in one
      terminal, with no external supervisor.

  KEPT (everything that actually matters):
    - BC warm-start loading with corrupt-zip recovery (fixed: replace() not rename())
    - CurriculumCallback  —  spawn frontier advances as the model improves
    - RaceRecordCallback  —  per-phase JSONL logging + race_records.json update
    - auto_backup         —  timestamped backups whenever avg_gates hits a new high
    - safe_save           —  atomic tmp→rename so Ctrl-C never corrupts the model
    - watch_episode / LiveWatchCallback  —  optional real-time OpenCV window
    - eval_only           —  evaluate without training

Usage
-----
    # Start / resume training (loads BC warm-start or last checkpoint automatically)
    python train.py --vision

    # One phase then stop
    python train.py --vision --steps 100000 --phases 1

    # Evaluate current best model
    python train.py --vision --eval-only

    # Watch a live episode every 25k steps while training
    python train.py --vision --watch-freq 25000

    # Write to a custom log file
    python train.py --vision --log-file logs/my_run.jsonl
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

# Force UTF-8 output on Windows (avoids cp1252 errors with box/arrow chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from simulation.drone_gym_env import RacingEnv  # noqa

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = pathlib.Path(__file__).parent
RECORDS_FILE = ROOT / "records" / "race_records.json"
MODEL_DIR    = ROOT / "trained_models"
LOG_DIR      = ROOT / "logs" / "tensorboard"
BACKUP_DIR   = MODEL_DIR / "backups"

# GPS-obs checkpoints (unchanged — interoperable with self_train.py)
CHECKPOINT   = MODEL_DIR / "rl_policy_latest"
BEST_MODEL   = MODEL_DIR / "rl_policy_best"

# Vision checkpoints (unchanged — interoperable with self_train.py)
VISION_CHECKPOINT = MODEL_DIR / "rl_vision_policy_latest"
VISION_BEST       = MODEL_DIR / "rl_vision_policy_best"

# Default training log (separate from self_train.py log to avoid interleaving)
VISION_TRAIN_LOG = ROOT / "records" / "vision_training_log.jsonl"
GPS_TRAIN_LOG    = ROOT / "records" / "training_log.jsonl"

for _d in (MODEL_DIR, LOG_DIR, BACKUP_DIR, ROOT / "records"):
    _d.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TRAIN_STEPS   = 100_000
EVAL_EPISODES = 8
N_ENVS        = 12
N_ENVS_VISION = 12
N_TRACK_GATES = 5
MAX_BACKUPS   = 20

PPO_PARAMS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 512,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.02,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    verbose       = 1,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_records() -> dict:
    if RECORDS_FILE.exists():
        try:
            return json.loads(RECORDS_FILE.read_text())
        except Exception:
            pass
    return {'best_time': None, 'best_gates': 0, 'best_date': None,
            'attempts': 0, 'best_splits': [], 'history': []}


def save_records(rec: dict) -> None:
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_FILE.write_text(json.dumps(rec, indent=2))


def make_env(rank: int = 0, use_camera_obs: bool = False):
    def _init():
        return RacingEnv(substeps=4, use_camera_obs=use_camera_obs)
    return _init


def _unwrap(env):
    while hasattr(env, 'env'):
        env = env.env
    return env


def safe_save(model: PPO, path: pathlib.Path) -> None:
    """Atomic tmp → rename save.  Retries for 5s if destination is locked."""
    tmp = path.with_suffix('.tmp')
    model.save(str(tmp.with_suffix('')))
    tmp_zip = tmp.with_suffix('.zip')
    src = tmp_zip if tmp_zip.exists() else tmp
    dst = path.with_suffix('.zip')
    for attempt in range(10):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == 0:
                print('  [safe_save] destination locked, retrying…', flush=True)
            time.sleep(0.5)
    src.replace(dst)


def auto_backup(model: PPO, label: str) -> None:
    """Timestamped backup whenever avg_gates hits a new high."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts         = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_label = label.replace('.', 'p')
    dst        = BACKUP_DIR / f"train_{safe_label}_{ts}.zip"
    model.save(str(dst.with_suffix('')))
    print(f'  [BACKUP] Saved backups/{dst.name}', flush=True)
    existing = sorted(BACKUP_DIR.glob('*.zip'), key=lambda p: p.stat().st_mtime)
    for old in existing[:-MAX_BACKUPS]:
        old.unlink(missing_ok=True)
        print(f'  [BACKUP] Pruned {old.name}', flush=True)


def build_or_load_model(env, checkpoint: pathlib.Path) -> PPO:
    """Return a fresh PPO or continue from checkpoint.

    If the zip is corrupt it is moved aside (never causes a crash-loop) and a
    fresh model is built.  Uses replace() instead of rename() so it works on
    Windows even when a previous .corrupt file already exists.
    """
    import zipfile
    ckpt = checkpoint.with_suffix('.zip')
    if ckpt.exists():
        try:
            with zipfile.ZipFile(ckpt) as zf:
                zf.testzip()
            print(f'[TRAIN] Loading checkpoint: {ckpt}')
            model = PPO.load(str(checkpoint), env=env,
                             tensorboard_log=str(LOG_DIR))
            return model
        except (zipfile.BadZipFile, ValueError, Exception) as e:
            corrupt = ckpt.with_suffix('.corrupt')
            corrupt.unlink(missing_ok=True)   # clear any pre-existing .corrupt
            ckpt.replace(corrupt)             # atomic on all platforms
            print(f'[TRAIN] WARNING: checkpoint corrupt ({e})')
            print(f'        Moved to {corrupt.name} — starting fresh.')

    print('[TRAIN] No valid checkpoint — building fresh PPO model.')
    return PPO(policy='MlpPolicy', env=env,
               tensorboard_log=str(LOG_DIR), **PPO_PARAMS)


# ── Callbacks ─────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Advance spawn gate as the model consistently reaches each gate."""
    ADVANCE_THRESHOLD = 0.10
    ADVANCE_PATIENCE  = 2
    CHECK_EVERY       = 8_192
    FRONTIER_RATIO    = 0.5

    def __init__(self, raw_envs: list, n_gates: int = N_TRACK_GATES, verbose: int = 1):
        super().__init__(verbose)
        self._raw_envs        = raw_envs
        self._n_gates         = n_gates
        self._curriculum_gate = 0
        self._patience        = 0
        self._last_check      = 0
        self._ep_gates: list  = []

    @property
    def curriculum_gate(self) -> int:
        return self._curriculum_gate

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get('dones', []),
                              self.locals.get('infos', [])):
            if done:
                self._ep_gates.append(int(info.get('gates_passed', 0)))
        if (self.num_timesteps - self._last_check) >= self.CHECK_EVERY:
            self._last_check = self.num_timesteps
            self._maybe_advance()
        return True

    def _maybe_advance(self) -> None:
        if len(self._ep_gates) < 4:
            return
        recent = self._ep_gates[-40:]
        avg    = sum(recent) / len(recent)
        if avg >= self._curriculum_gate + self.ADVANCE_THRESHOLD:
            self._patience += 1
        else:
            self._patience = max(0, self._patience - 1)

        if (self._patience >= self.ADVANCE_PATIENCE
                and self._curriculum_gate < self._n_gates - 2):
            prev = self._curriculum_gate
            self._curriculum_gate += 1
            self._patience = 0
            self._apply_spawn()
            print(f'\n  [CURRICULUM] ↑ Spawn → gate {self._curriculum_gate}'
                  f'  (avg: {avg:.2f}, was gate {prev})', flush=True)

    def _apply_spawn(self) -> None:
        n        = len(self._raw_envs)
        frontier = max(1, int(n * self.FRONTIER_RATIO))
        for i, env in enumerate(self._raw_envs):
            _unwrap(env).curriculum_gate = (
                self._curriculum_gate if i < frontier else 0)


class LiveWatchCallback(BaseCallback):
    """Pop up a rendered episode every `watch_every` training steps."""
    def __init__(self, watch_every: int = 25_000, verbose: int = 0):
        super().__init__(verbose)
        self._watch_every  = watch_every
        self._last_watched = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_watched) >= self._watch_every:
            self._last_watched = self.num_timesteps
            print(f'\n  [WATCH] {self.num_timesteps/1000:.0f}k steps — '
                  f'opening live window…')
            watch_episode(self.model)
        return True


class RaceRecordCallback(BaseCallback):
    """Per-phase evaluation, JSONL logging, and best-model saving."""

    def __init__(self, eval_env, n_eval: int = EVAL_EPISODES,
                 prefix: str = 'rl_vision',
                 log_file: pathlib.Path = VISION_TRAIN_LOG,
                 checkpoint: pathlib.Path = VISION_CHECKPOINT,
                 best_path: pathlib.Path = VISION_BEST,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env   = eval_env
        self.n_eval     = n_eval
        self._prefix    = prefix
        self._log       = log_file
        self._ckpt      = checkpoint
        self._best      = best_path
        self._phase     = 0
        self._best_avg  = 0.0

    def _on_training_end(self) -> None:
        self._phase += 1
        print(f'\n[EVAL] Phase {self._phase} — {self.n_eval} episodes…')

        times, gates_list = [], []
        for ep in range(self.n_eval):
            obs, _ = self.eval_env.reset()
            done   = False
            g      = 0
            t0     = time.time()
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                g    = info.get('gates_passed', g)
            times.append(time.time() - t0)
            gates_list.append(g)
            print(f'  Ep {ep+1}: {g}/{N_TRACK_GATES} gates  {times[-1]:.2f}s')

        avg_gates   = sum(gates_list) / max(len(gates_list), 1)
        best_ep     = max(gates_list)
        clean       = [t for t, g in zip(times, gates_list) if g >= N_TRACK_GATES]
        best_clean  = min(clean) if clean else None

        # Auto-backup on genuine improvement
        if avg_gates > self._best_avg + 0.05:
            self._best_avg = avg_gates
            auto_backup(self.model, f'avg{avg_gates:.1f}_ph{self._phase}')
            safe_save(self.model, self._best)
            print(f'  [BEST] avg_gates={avg_gates:.1f} new high → {self._best.name}.zip',
                  flush=True)

        records     = load_records()
        scripted    = records.get('best_time')
        rl_record   = records.get(f'{self._prefix}_best_time')
        prev_avg    = records.get(f'{self._prefix}_last_avg_gates', 0.0)

        if avg_gates > prev_avg + 0.1:
            trend = f'↑ +{avg_gates - prev_avg:.1f}'
        elif avg_gates < prev_avg - 0.1:
            trend = f'↓ {avg_gates - prev_avg:.1f}'
        else:
            trend = '→ stable'

        print(f'  Avg gates: {avg_gates:.1f}/{N_TRACK_GATES}  '
              f'best_ep: {best_ep}/{N_TRACK_GATES}  {trend}')

        if scripted:
            if best_clean:
                d = best_clean - scripted
                lbl = 'BEATS SCRIPTED' if d < 0 else f'{d:+.2f}s behind scripted'
                print(f'  Best clean: {best_clean:.2f}s  Scripted: {scripted:.2f}s  [{lbl}]')
            else:
                print(f'  No clean finish.  Scripted: {scripted:.2f}s')
        elif best_clean:
            print(f'  Best clean: {best_clean:.2f}s')
        else:
            print('  No clean finish yet.')

        # Update records
        new_record = False
        if best_clean is not None and (rl_record is None or best_clean < rl_record):
            records[f'{self._prefix}_best_time']  = round(best_clean, 3)
            records[f'{self._prefix}_best_gates'] = N_TRACK_GATES
            records[f'{self._prefix}_best_date']  = datetime.date.today().isoformat()
            new_record = True
        if best_clean is not None:
            curr = records.get('best_time')
            if curr is None or best_clean < curr:
                records['best_time']  = round(best_clean, 3)
                records['best_gates'] = N_TRACK_GATES
                records['best_date']  = datetime.date.today().isoformat()

        records[f'{self._prefix}_last_avg_gates'] = round(avg_gates, 3)
        records.setdefault('history', []).append({
            'phase': self._phase,
            'total_steps': int(self.num_timesteps),
            'avg_gates': round(avg_gates, 3),
            'best_gates_ep': int(best_ep),
            'best_clean_time': round(best_clean, 3) if best_clean else None,
            'date': datetime.date.today().isoformat(),
            'source': 'train.py',
        })
        records['history'] = records['history'][-100:]
        save_records(records)

        if new_record:
            safe_save(self.model, self._ckpt)
            print(f'  [NEW RECORD]  {best_clean:.2f}s')

        # JSONL log
        try:
            with open(self._log, 'a') as f:
                f.write(json.dumps({
                    'ts':            datetime.datetime.now().isoformat(timespec='seconds'),
                    'phase':         self._phase,
                    'total_steps':   int(self.num_timesteps),
                    'avg_gates':     round(avg_gates, 3),
                    'best_gates_ep': int(best_ep),
                    'best_clean_s':  round(best_clean, 3) if best_clean else None,
                    'scripted_s':    scripted,
                    'new_record':    new_record,
                    'trend':         trend,
                }) + '\n')
        except Exception as e:
            print(f'  [WARN] Could not write log: {e}')
        print()

    def _on_step(self) -> bool:
        return True


# ── Core training function ────────────────────────────────────────────────────

def train(total_steps: int = TRAIN_STEPS,
          vision: bool = True,
          render: bool = False,
          watch_freq: int = 0,
          max_phases: int = 0,
          log_file: pathlib.Path = None) -> None:
    """
    Run the PPO training loop indefinitely (or for `max_phases` phases).
    No lock file, no watchdog hooks — just train.
    """
    n_envs     = N_ENVS_VISION if vision else N_ENVS
    checkpoint = VISION_CHECKPOINT if vision else CHECKPOINT
    best_path  = VISION_BEST if vision else BEST_MODEL
    prefix     = 'rl_vision' if vision else 'rl'
    if log_file is None:
        log_file = VISION_TRAIN_LOG if vision else GPS_TRAIN_LOG
    mode_label = 'VISION (camera obs, 110° FOV)' if vision else 'GPS state obs'

    print('=' * 65)
    print('  DRONEMANIA — TRAINING LOOP  (train.py — no watchdog)')
    print(f'  Mode: {mode_label}')
    print(f'  {n_envs} envs × {total_steps:,} steps/phase | PPO (MlpPolicy)')
    if max_phases:
        print(f'  Max phases: {max_phases}')
    print('=' * 65)

    dummy_envs = DummyVecEnv([make_env(i, use_camera_obs=vision) for i in range(n_envs)])
    train_envs = VecMonitor(dummy_envs)
    eval_env   = RacingEnv(substeps=4, use_camera_obs=vision)

    model      = build_or_load_model(train_envs, checkpoint)
    curriculum = CurriculumCallback(dummy_envs.envs, N_TRACK_GATES)
    record_cb  = RaceRecordCallback(
        eval_env, n_eval=EVAL_EPISODES, prefix=prefix,
        log_file=log_file, checkpoint=checkpoint, best_path=best_path)

    cb_list = [curriculum, record_cb]
    if watch_freq > 0:
        cb_list.append(LiveWatchCallback(watch_every=watch_freq))
    callback = CallbackList(cb_list)

    phase = 0
    try:
        while True:
            phase += 1
            print(f'\n{"─"*65}')
            print(f'  Training phase {phase}  ({total_steps:,} steps)  [{mode_label}]')
            print(f'{"─"*65}')
            model.learn(total_timesteps=total_steps,
                        callback=callback,
                        reset_num_timesteps=(phase == 1),
                        progress_bar=True)
            safe_save(model, checkpoint)
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'\n[TRAIN] {ts} | Phase {phase} complete | '
                  f'checkpoint → {checkpoint.name}.zip')
            print(f'  [CURRICULUM] Spawn gate: '
                  f'{curriculum.curriculum_gate}/{N_TRACK_GATES-1}')
            if render:
                watch_episode(model, vision=vision)
            if max_phases and phase >= max_phases:
                print(f'\n[TRAIN] Reached max_phases={max_phases} — stopping.')
                break

    except KeyboardInterrupt:
        print('\n[TRAIN] Ctrl-C — saving…')
        safe_save(model, checkpoint)
        print(f'  Saved: {checkpoint}.zip')
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(f'\n[TRAIN] CRASH in phase {phase}:\n{tb}', flush=True)
        crash = ROOT / 'logs' / 'crash.txt'
        crash.parent.mkdir(exist_ok=True)
        crash.write_text(tb)
        print(f'  Traceback → {crash}')
        raise
    finally:
        train_envs.close()
        eval_env.close()


# ── Eval-only ─────────────────────────────────────────────────────────────────

def watch_episode(model: PPO, vision: bool = False) -> None:
    try:
        import cv2
    except ImportError:
        print('  [WATCH] opencv-python not installed — skipping.')
        return
    env  = RacingEnv(substeps=4, render_mode='human', use_camera_obs=vision)
    obs, _ = env.reset()
    env.render()
    done = False; gates = 0; aborted = False
    print('  [WATCH] Press Q in window to skip…')
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done  = terminated or truncated
        gates = info.get('gates_passed', gates)
        result = env.render()
        if isinstance(result, str) and result == 'quit':
            aborted = True; break
    env.close()
    try: cv2.destroyAllWindows()
    except Exception: pass
    msg = f'{gates}/{N_TRACK_GATES} gates'
    print(f'  [WATCH] {"Skipped" if aborted else "Finished"} — {msg}')


def eval_only(vision: bool = False, render: bool = False, n: int = 10) -> None:
    path = (VISION_BEST if vision else BEST_MODEL).with_suffix('.zip')
    if not path.exists():
        path = (VISION_CHECKPOINT if vision else CHECKPOINT).with_suffix('.zip')
    if not path.exists():
        print(f'[EVAL] No model found for {"vision" if vision else "GPS"} mode.')
        return
    rmode = 'human' if render else None
    env   = RacingEnv(substeps=4, render_mode=rmode, use_camera_obs=vision)
    model = PPO.load(str(path.with_suffix('')), env=env)
    print(f'Loaded: {path}')
    for ep in range(n):
        obs, _ = env.reset()
        done   = False; g = 0; t0 = time.time()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            g    = info.get('gates_passed', g)
            if render:
                result = env.render()
                if isinstance(result, str) and result == 'quit': break
        print(f'Ep {ep+1:2d}: {g}/{N_TRACK_GATES} gates  {time.time()-t0:.2f}s')
    env.close()
    try:
        import cv2; cv2.destroyAllWindows()
    except ImportError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Dronemania training — no watchdog')
    p.add_argument('--vision',     action='store_true',
                   help='Train with camera observations instead of GPS state')
    p.add_argument('--steps',      type=int, default=TRAIN_STEPS,
                   help='PPO steps per phase (default: 100,000)')
    p.add_argument('--phases',     type=int, default=0,
                   help='Stop after this many phases (0 = run forever)')
    p.add_argument('--eval-only',  action='store_true',
                   help='Evaluate current best model without training')
    p.add_argument('--eval-n',     type=int, default=10,
                   help='Number of eval episodes for --eval-only')
    p.add_argument('--render',     action='store_true',
                   help='Show live window after each phase (or every episode with --eval-only)')
    p.add_argument('--watch-freq', type=int, default=0, metavar='STEPS',
                   help='Show live window every N steps during training (0 = off)')
    p.add_argument('--log-file',   type=pathlib.Path, default=None,
                   help='Override JSONL log file path')
    args = p.parse_args()

    if args.eval_only:
        eval_only(vision=args.vision, render=args.render, n=args.eval_n)
    else:
        train(total_steps=args.steps, vision=args.vision, render=args.render,
              watch_freq=args.watch_freq, max_phases=args.phases,
              log_file=args.log_file)
