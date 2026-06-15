#!/usr/bin/env python3
"""
self_train.py — Autonomous self-improving training loop.

Architecture
============
This is a HYBRID system:

  Layer 1 – Scripted Baseline (visualize_race.py)
      The hand-tuned PD controller that already achieves 5/5 gates.
      Used as the "reference" whose lap times we try to beat.

  Layer 2 – Learned Policy (PPO via stable-baselines3)
      Trains inside RacingEnv (the real RacingSimulator + DronePhysics).
      Observation: 20-float state vector (position, velocity, gates, …)
      Action: normalised velocity commands → inner PD motor loop.
      The RL agent learns WHEN and HOW FAST to move — the low-level
      attitude stabilisation is still handled by the same PD controller
      inside the gym env, so the two layers don't conflict.

  No framework overlap
      Adding ROS2/PX4/MAVSDK here would add hardware-protocol overhead
      with zero benefit for a pure-Python simulation.  Gymnasium + SB3
      is the correct autonomy abstraction for this stack.

Self-improving loop
===================
  1. Load existing checkpoint (if any) so every run builds on the last.
  2. Train for TRAIN_STEPS steps.
  3. Run EVAL_EPISODES evaluation episodes; record best completion time.
  4. Compare against records/race_records.json (set by visualize_race.py).
  5. If RL beats the scripted record → overwrite with new best + save checkpoint.
  6. Repeat from step 2 indefinitely (Ctrl-C to stop cleanly).

Usage
-----
    python self_train.py                  # train forever
    python self_train.py --steps 50000    # one training phase then exit
    python self_train.py --eval-only      # evaluate current checkpoint only
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

# ── Add src to path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                 EvalCallback)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from simulation.drone_gym_env import RacingEnv  # noqa

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = pathlib.Path(__file__).parent
RECORDS_FILE  = ROOT / "records" / "race_records.json"
MODEL_DIR     = ROOT / "trained_models"
LOG_DIR       = ROOT / "logs" / "tensorboard"
CHECKPOINT    = MODEL_DIR / "rl_policy_latest"
BEST_MODEL    = MODEL_DIR / "rl_policy_best"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "records").mkdir(parents=True, exist_ok=True)

# Vision-mode checkpoint paths (separate from the GPS-obs model)
VISION_CHECKPOINT = MODEL_DIR / "rl_vision_policy_latest"
VISION_BEST       = MODEL_DIR / "rl_vision_policy_best"
VISION_TRAIN_LOG  = ROOT / "records" / "vision_training_log.jsonl"

# Timestamped backup directory — never auto-deleted, max MAX_BACKUPS kept
BACKUP_DIR  = MODEL_DIR / "backups"
MAX_BACKUPS = 20
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────
TRAIN_STEPS    = 100_000   # steps per training phase
EVAL_EPISODES  = 16        # bumped 8→16: more stable avg metric, reduces lucky/unlucky track variance
N_ENVS         = 12        # parallel environments (GPS mode)
N_ENVS_VISION  = 12        # vision mode: 12 envs targets ~75% CPU on 4-core machine
N_TRACK_GATES  = 5         # number of gates on the track (matches racing_simulator default)
PPO_PARAMS = dict(
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 512,   # scaled up: 12 envs × 2048 steps = 24,576 / 512 = 48 minibatches
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,    # lowered 0.05→0.01: gate-2 plateau broken, now exploit stable policy
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
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
    """Factory for a single RacingEnv (VecMonitor is applied at the vec level)."""
    def _init():
        env = RacingEnv(substeps=4, use_camera_obs=use_camera_obs)
        return env
    return _init


def _unwrap(env):
    """Unwrap a Monitor/TimeLimit wrapper to reach the raw RacingEnv."""
    while hasattr(env, 'env'):
        env = env.env
    return env


def safe_save(model: PPO, path: pathlib.Path) -> None:
    """Save model to a .tmp file then atomically rename — Ctrl-C safe.

    Retries the rename for up to 5s if the destination is locked by another
    process (e.g. the monitor having the zip open on Windows).
    """
    tmp = path.with_suffix('.tmp')
    model.save(str(tmp.with_suffix('')))   # SB3 appends .zip itself
    tmp_zip = tmp.with_suffix('.zip')
    src = tmp_zip if tmp_zip.exists() else tmp
    dst = path.with_suffix('.zip')
    for attempt in range(10):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == 0:
                print(f"  [safe_save] destination locked, retrying...", flush=True)
            time.sleep(0.5)
    # Last resort: overwrite directly (not atomic but better than crashing)
    src.replace(dst)


def auto_backup(model: PPO, label: str) -> None:
    """Save a timestamped model snapshot to trained_models/backups/.

    Called automatically whenever avg_gates hits a new high so no progress
    is ever silently lost.  Keeps the newest MAX_BACKUPS files.
    """
    import shutil as _sh
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Avoid dots in the stem — SB3 uses endswith('.zip') to decide whether to
    # append the extension, so a label like "avg2.0" would confuse it.
    safe_label = label.replace('.', 'p')
    dst = BACKUP_DIR / f"vision_{safe_label}_{ts}.zip"
    model.save(str(dst.with_suffix('')))   # SB3 will re-append .zip
    print(f"  [BACKUP] Saved backups/{dst.name}", flush=True)
    # Prune: keep only the newest MAX_BACKUPS files
    existing = sorted(BACKUP_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    for old in existing[:-MAX_BACKUPS]:
        old.unlink(missing_ok=True)
        print(f"  [BACKUP] Pruned {old.name}", flush=True)


def build_or_load_model(env, checkpoint: pathlib.Path = CHECKPOINT) -> PPO:
    """Return a fresh PPO or continue from the latest checkpoint.

    Validates the zip file before loading — a corrupted checkpoint (from a
    mid-save Ctrl-C) is renamed to .corrupt and training starts fresh so the
    process never crashes on startup.
    """
    import zipfile
    ckpt = checkpoint.with_suffix('.zip')
    if ckpt.exists():
        # Validate before loading
        try:
            with zipfile.ZipFile(ckpt) as zf:
                zf.testzip()   # raises BadZipFile if corrupt
            print(f"[SELF-TRAIN] Loading checkpoint: {ckpt}")
            model = PPO.load(str(checkpoint), env=env,
                             tensorboard_log=str(LOG_DIR))
            return model
        except (zipfile.BadZipFile, ValueError, Exception) as e:
            corrupt = ckpt.with_suffix('.corrupt')
            corrupt.unlink(missing_ok=True)   # Windows rename() won't overwrite; clear first
            ckpt.replace(corrupt)             # replace() is atomic and works on all platforms
            print(f"[SELF-TRAIN] WARNING: checkpoint was corrupt ({e})")
            print(f"             Renamed to {corrupt.name} — starting fresh.")

    print("[SELF-TRAIN] Starting fresh PPO model  (MlpPolicy, 20-dim obs)")
    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        tensorboard_log = str(LOG_DIR),
        **PPO_PARAMS
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

TRAIN_LOG = ROOT / "records" / "training_log.jsonl"   # one JSON line per eval phase


# ── Trainer lock (prevents duplicate instances) ───────────────────────────────
_TRAINER_LOCK = MODEL_DIR / "trainer.lock"


def _pid_alive(pid: int) -> bool:
    """Return True if the given PID is still running (Windows-compatible)."""
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong(0)
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == 259   # 259 = STILL_ACTIVE
    except Exception:
        return False  # assume stale on any error


def _acquire_lock() -> bool:
    """
    Write this process's PID to _TRAINER_LOCK.
    Returns False if another live trainer already holds the lock.
    """
    if _TRAINER_LOCK.exists():
        try:
            other = int(_TRAINER_LOCK.read_text().strip())
            if _pid_alive(other):
                print(f"[SELF-TRAIN] Another trainer already running (PID {other}).")
                print(f"  Delete {_TRAINER_LOCK} to override.")
                return False
        except Exception:
            pass  # stale / unreadable lock — overwrite it
    _TRAINER_LOCK.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    """Remove the trainer lock file (called in finally so it always runs)."""
    try:
        _TRAINER_LOCK.unlink(missing_ok=True)
    except Exception:
        pass


class CurriculumCallback(BaseCallback):
    """
    Tracks which gate the model is consistently reaching and gradually moves the
    spawn point forward — so training always targets the learning frontier.

    Advancement rule
    ----------------
    Every CHECK_EVERY steps, compute rolling mean of max-gates-per-episode.
    If that mean exceeds (curriculum_gate + ADVANCE_THRESHOLD) for
    ADVANCE_PATIENCE consecutive checks, advance curriculum_gate by 1.

    Spawn mix
    ---------
    FRONTIER_RATIO of envs spawn at the frontier gate.
    The rest spawn at gate 0 so the policy never forgets the start.
    """
    ADVANCE_THRESHOLD = 0.10   # gates above frontier before advancing (lowered from 0.25)
    ADVANCE_PATIENCE  = 2      # consecutive checks required (was 3)
    CHECK_EVERY       = 8_192  # steps between checks (one PPO rollout)
    FRONTIER_RATIO    = 0.5    # fraction of envs at curriculum gate
    ROLLBACK_THRESHOLD= 0.3    # if performance drops this much, don't regress

    def __init__(self, raw_envs: list, n_gates: int = N_TRACK_GATES, verbose: int = 1):
        super().__init__(verbose)
        self._raw_envs        = raw_envs   # list of actual RacingEnv objects
        self._n_gates         = n_gates
        self._curriculum_gate = 0
        self._patience        = 0
        self._last_check      = 0
        self._ep_gates: list  = []         # max gates per finished episode

    @property
    def curriculum_gate(self) -> int:
        return self._curriculum_gate

    def _on_step(self) -> bool:
        # Collect gates_passed whenever an episode finishes in any env
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        for done, info in zip(dones, infos):
            if done:
                self._ep_gates.append(int(info.get('gates_passed', 0)))

        if (self.num_timesteps - self._last_check) >= self.CHECK_EVERY:
            self._last_check = self.num_timesteps
            self._maybe_advance()
        return True

    def _maybe_advance(self) -> None:
        if len(self._ep_gates) < 4:
            return
        recent = self._ep_gates[-40:]   # rolling window
        avg    = sum(recent) / len(recent)
        target = self._curriculum_gate + self.ADVANCE_THRESHOLD

        if avg >= target:
            self._patience += 1
        else:
            self._patience = max(0, self._patience - 1)

        adv = self._curriculum_gate
        if (self._patience >= self.ADVANCE_PATIENCE
                and self._curriculum_gate < self._n_gates - 2):
            self._curriculum_gate += 1
            self._patience = 0
            self._apply_spawn()
            print(f"\n  [CURRICULUM] \u2191 Spawn → gate {self._curriculum_gate}  "
                  f"(avg recent: {avg:.2f} gates, was targeting gate {adv})")
        elif self.verbose >= 2:
            print(f"  [CURRICULUM] gate {self._curriculum_gate}  "
                  f"avg={avg:.2f}  patience={self._patience}/{self.ADVANCE_PATIENCE}")

    def _apply_spawn(self) -> None:
        """Push curriculum_gate into the underlying RacingEnv instances."""
        n        = len(self._raw_envs)
        frontier = max(1, int(n * self.FRONTIER_RATIO))
        for i, env in enumerate(self._raw_envs):
            raw = _unwrap(env)
            raw.curriculum_gate = self._curriculum_gate if i < frontier else 0


class LiveWatchCallback(BaseCallback):
    """
    Every `watch_every` training steps, pause and run one rendered episode so
    you can see how the current policy is performing without waiting for the
    end of a full training phase.

    The watch window is a regular OpenCV window — press Q to skip the episode
    and return to training immediately.
    """
    def __init__(self, watch_every: int = 25_000, verbose: int = 0):
        super().__init__(verbose)
        self._watch_every  = watch_every
        self._last_watched = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_watched) >= self._watch_every:
            self._last_watched = self.num_timesteps
            steps_k = self.num_timesteps / 1000
            print(f"\n  [WATCH] {steps_k:.0f}k steps — opening live window …")
            watch_episode(self.model)
        return True


class RaceRecordCallback(BaseCallback):
    """
    After each training phase completes:
      - Run EVAL_EPISODES evaluation episodes (deterministic policy)
      - Log results to records/training_log.jsonl (one line per phase)
      - Update records/race_records.json if a new best time is set
      - Save best model weights to trained_models/rl_policy_best.zip

    The JSONL log gives the model cognizance of its own history so each phase
    can be compared against:
      • Its own previous phases (are we improving?)
      • The scripted PD baseline (are we competitive?)
    """
    def __init__(self, eval_env, n_eval: int = EVAL_EPISODES, verbose: int = 1,
                 record_prefix: str = 'rl', train_log: pathlib.Path = TRAIN_LOG):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval   = n_eval
        self._phase   = 0
        self._prefix  = record_prefix   # 'rl' or 'rl_vision'
        self._log     = train_log
        self._best_backup_avg = 0.0     # tracks highest avg_gates seen; only backs up on genuine improvements

    # ── called once at the end of every model.learn() call ───────────────────
    def _on_training_end(self) -> None:
        self._phase += 1
        print(f"\n[EVAL] Phase {self._phase} — running {self.n_eval} evaluation episodes …")

        times, gates_list = [], []
        for ep in range(self.n_eval):
            obs, _ = self.eval_env.reset()
            done     = False
            ep_gates = 0
            t_start  = time.time()
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done     = terminated or truncated
                ep_gates = info.get('gates_passed', ep_gates)
            elapsed = time.time() - t_start
            times.append(elapsed)
            gates_list.append(ep_gates)
            print(f"  Ep {ep+1}: {ep_gates}/{N_TRACK_GATES} gates  {elapsed:.2f}s")

        avg_gates = sum(gates_list) / max(len(gates_list), 1)
        best_gates_ep = max(gates_list)
        clean = [t for t, g in zip(times, gates_list) if g >= N_TRACK_GATES]
        best_clean = min(clean) if clean else None

        # ── Auto-backup whenever avg_gates climbs to a new personal best ──────
        if avg_gates > self._best_backup_avg + 0.05:
            self._best_backup_avg = avg_gates
            auto_backup(self.model, f"avg{avg_gates:.1f}_ph{self._phase}")
            # Also keep a stable recovery checkpoint for cleanup.ps1
            best_ckpt = VISION_BEST if self._prefix == 'rl_vision' else BEST_MODEL
            safe_save(self.model, best_ckpt)
            print(f"  [BEST] avg_gates={avg_gates:.1f} new high → saved {best_ckpt.name}.zip", flush=True)

        records = load_records()
        scripted_best = records.get('best_time')                         # set by visualize_race.py
        rl_record     = records.get(f'{self._prefix}_best_time')         # this model's record
        prev_avg      = records.get(f'{self._prefix}_last_avg_gates', 0.0)

        # ── Determine trend vs previous phase ────────────────────────────────
        if avg_gates > prev_avg + 0.1:
            trend = f"↑ +{avg_gates - prev_avg:.1f} gates avg"
        elif avg_gates < prev_avg - 0.1:
            trend = f"↓ {avg_gates - prev_avg:.1f} gates avg"
        else:
            trend = "→ stable"

        print(f"  Avg gates: {avg_gates:.1f}/{N_TRACK_GATES}  best_ep: {best_gates_ep}/{N_TRACK_GATES}  {trend}")

        # ── Compare against scripted baseline ─────────────────────────────────
        if scripted_best:
            if best_clean:
                delta = best_clean - scripted_best
                label = ("BEATS SCRIPTED" if delta < 0
                         else f"{delta:+.2f}s behind scripted")
                print(f"  Best clean: {best_clean:.2f}s  |  Scripted: {scripted_best:.2f}s  [{label}]")
            else:
                print(f"  No clean finish yet.  Scripted baseline: {scripted_best:.2f}s")
        elif best_clean:
            print(f"  Best clean: {best_clean:.2f}s  (no scripted baseline on record yet)")
        else:
            print(f"  No clean finish yet.")

        # ── Update records ────────────────────────────────────────────────────
        new_rl_record = False
        if best_clean is not None and (rl_record is None or best_clean < rl_record):
            records[f'{self._prefix}_best_time']  = round(best_clean, 3)
            records[f'{self._prefix}_best_gates'] = N_TRACK_GATES
            records[f'{self._prefix}_best_date']  = datetime.date.today().isoformat()
            new_rl_record = True

        # Also update top-level best if this model beats the scripted time
        if best_clean is not None:
            curr_best = records.get('best_time')
            if curr_best is None or best_clean < curr_best:
                records['best_time']  = round(best_clean, 3)
                records['best_gates'] = N_TRACK_GATES
                records['best_date']  = datetime.date.today().isoformat()

        records[f'{self._prefix}_last_avg_gates'] = round(avg_gates, 3)
        records.setdefault('history', []).append({
            'phase': self._phase,
            'total_steps': int(self.num_timesteps),
            'avg_gates': round(avg_gates, 3),
            'best_gates_ep': int(best_gates_ep),
            'best_clean_time': round(best_clean, 3) if best_clean else None,
            'date': datetime.date.today().isoformat(),
            'source': 'rl_policy',
        })
        records['history'] = records['history'][-100:]
        save_records(records)

        if new_rl_record:
            safe_save(self.model, BEST_MODEL)
            prev_rl = records.get('rl_best_time')  # already updated above
            print(f"  [NEW RL RECORD]  {best_clean:.2f}s  — best model saved → {BEST_MODEL}.zip")

        # ── Append one line to JSONL training log ─────────────────────────────
        log_line = {
            'ts': datetime.datetime.now().isoformat(timespec='seconds'),
            'phase': self._phase,
            'total_steps': int(self.num_timesteps),
            'avg_gates': round(avg_gates, 3),
            'best_gates_ep': int(best_gates_ep),
            'best_clean_s': round(best_clean, 3) if best_clean else None,
            'scripted_best_s': scripted_best,
            'rl_record_s': records.get('rl_best_time'),
            'new_rl_record': new_rl_record,
            'trend': trend,
        }
        try:
            with open(self._log, 'a') as f:
                f.write(json.dumps(log_line) + '\n')
        except Exception as e:
            print(f"  [WARN] Could not write training log: {e}")
        print()

    def _on_step(self) -> bool:
        return True   # never stop training early from this callback



# ── Main ──────────────────────────────────────────────────────────────────────

def train(total_steps: int, render: bool = False, watch_freq: int = 0,
          vision: bool = False) -> None:
    # ── Duplicate-instance guard ──────────────────────────────────────────────
    if not _acquire_lock():
        sys.exit(99)  # code 99 = another trainer live; watchdog will back off

    n_envs     = N_ENVS_VISION if vision else N_ENVS
    checkpoint = VISION_CHECKPOINT if vision else CHECKPOINT
    prefix     = 'rl_vision' if vision else 'rl'
    log_file   = VISION_TRAIN_LOG if vision else TRAIN_LOG
    mode_label = 'VISION (camera obs, 110° FOV)' if vision else 'GPS state obs'

    print("=" * 65)
    print("  DRONEMANIA — SELF-TRAINING RL LOOP")
    print(f"  Mode: {mode_label}")
    print(f"  {n_envs} envs × {total_steps:,} steps/phase | PPO (MlpPolicy)")
    if render:
        print("  Live watch: ON — one rendered episode after each phase")
    if watch_freq > 0:
        print(f"  Live watch mid-training: every {watch_freq:,} steps")
    print("=" * 65)

    # Training envs (vectorised) — keep reference to DummyVecEnv for curriculum
    dummy_envs = DummyVecEnv([make_env(i, use_camera_obs=vision) for i in range(n_envs)])
    train_envs = VecMonitor(dummy_envs)

    # Single eval env
    eval_env = RacingEnv(substeps=4, use_camera_obs=vision)

    model      = build_or_load_model(train_envs, checkpoint)
    curriculum = CurriculumCallback(dummy_envs.envs, N_TRACK_GATES)
    record_cb  = RaceRecordCallback(eval_env, n_eval=EVAL_EPISODES,
                                    record_prefix=prefix, train_log=log_file)

    from stable_baselines3.common.callbacks import CallbackList
    cb_list = [curriculum, record_cb]
    if watch_freq > 0:
        cb_list.append(LiveWatchCallback(watch_every=watch_freq))
    callback = CallbackList(cb_list)

    phase = 0
    try:
        while True:
            phase += 1
            print(f"\n{'─'*65}")
            print(f"  Training phase {phase}  ({total_steps:,} steps)  [{mode_label}]")
            print(f"{'─'*65}")
            model.learn(total_timesteps=total_steps,
                        callback=callback,
                        reset_num_timesteps=(phase == 1),
                        progress_bar=True)
            safe_save(model, checkpoint)
            _ts = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"\n[TRAINER→MONITOR] {_ts} | Phase {phase} complete | "
                  f"Checkpoint saved → {checkpoint.name}.zip")
            print(f"  [CURRICULUM] Current spawn gate: {curriculum.curriculum_gate}/{N_TRACK_GATES-1}")
            if render:
                watch_episode(model, vision=vision)

    except KeyboardInterrupt:
        print("\n[SELF-TRAIN] Interrupted — saving checkpoint …")
        safe_save(model, checkpoint)
        print(f"  Saved: {checkpoint}.zip")
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[SELF-TRAIN] CRASH in phase {phase}:\n{tb}", flush=True)
        log_dir = pathlib.Path("logs")
        log_dir.mkdir(exist_ok=True)
        crash_file = log_dir / "crash.txt"
        crash_file.write_text(tb)
        print(f"  Traceback written to {crash_file}")
        raise
    finally:
        _release_lock()   # always clean up lock regardless of how we exit

    train_envs.close()
    eval_env.close()


def watch_episode(model: PPO, vision: bool = False) -> None:
    """
    Play one episode with the live OpenCV window so you can watch the current
    policy race in real time.  Called after each training phase when --render.
    Press Q inside the window to skip to the next phase.
    """
    try:
        import cv2
    except ImportError:
        print("  [WATCH] opencv-python not installed — skipping render.")
        return

    env = RacingEnv(substeps=4, render_mode='human', use_camera_obs=vision)
    obs, _ = env.reset()
    env.render()   # open window once
    done    = False
    gates   = 0
    t_start = time.time()
    aborted = False
    print("  [WATCH] Live episode running … (press Q in the window to skip)")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done  = terminated or truncated
        gates = info.get('gates_passed', gates)
        result = env.render()
        if isinstance(result, str) and result == 'quit':
            aborted = True
            break
    elapsed = time.time() - t_start
    env.close()
    cv2.destroyAllWindows()
    if not aborted:
        print(f"  [WATCH] Finished — {gates}/{N_TRACK_GATES} gates  {elapsed:.2f}s")
    else:
        print(f"  [WATCH] Skipped — {gates}/{N_TRACK_GATES} gates so far")


def eval_only(render: bool = False, vision: bool = False) -> None:
    """Quick evaluation of the current best model."""
    path = (VISION_BEST if vision else BEST_MODEL).with_suffix('.zip')
    if not path.exists():
        path = (VISION_CHECKPOINT if vision else CHECKPOINT).with_suffix('.zip')
    if not path.exists():
        mode = 'vision' if vision else 'GPS'
        print(f"[EVAL-ONLY] No trained {mode} model found. Run training first.")
        return

    rmode = 'human' if render else None
    env   = RacingEnv(substeps=4, render_mode=rmode, use_camera_obs=vision)
    model = PPO.load(str(path.with_suffix('')), env=env)
    print(f"Loaded: {path}")
    if render:
        print("[EVAL-ONLY] Showing live window — press Q in window to skip episode")

    for ep in range(10):
        obs, _ = env.reset()
        done   = False
        gates  = 0
        t0     = time.time()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            gates = info.get('gates_passed', gates)
            if render:
                result = env.render()
                if isinstance(result, str) and result == 'quit':
                    break
        print(f"Ep {ep+1:2d}: {gates}/{N_TRACK_GATES} gates  {time.time()-t0:.2f}s")

    env.close()
    if render:
        try:
            import cv2; cv2.destroyAllWindows()
        except ImportError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dronemania self-training loop")
    parser.add_argument("--steps",     type=int,  default=TRAIN_STEPS,
                        help="PPO steps per training phase")
    parser.add_argument("--vision",     action="store_true",
                        help="Train/eval using visual camera observations instead of GPS state")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, no training")
    parser.add_argument("--fresh",     action="store_true",
                        help="Delete existing checkpoint and start training from scratch")
    parser.add_argument("--render", action="store_true",
                        help="Show live OpenCV window after each training phase (or every episode with --eval-only)")
    parser.add_argument("--watch-freq", type=int, default=0, metavar="STEPS",
                        help="Pop up a live window every N training steps (e.g. 25000). 0 = off.")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(render=args.render, vision=args.vision)
    else:
        if args.fresh:
            import shutil as _sh
            vision_paths  = [VISION_CHECKPOINT.with_suffix('.zip'), VISION_BEST.with_suffix('.zip')]
            gps_paths     = [CHECKPOINT.with_suffix('.zip'), BEST_MODEL.with_suffix('.zip')]
            targets       = vision_paths if args.vision else gps_paths
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            for p in targets:
                if p.exists():
                    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    bkup = BACKUP_DIR / f"pre_fresh_{p.stem}_{ts}.zip"
                    _sh.copy2(p, bkup)
                    print(f"[FRESH] Backed up → backups/{bkup.name}")
                    p.unlink()
                    print(f"[FRESH] Deleted {p.name}")
        train(args.steps, render=args.render, watch_freq=args.watch_freq,
              vision=args.vision)
