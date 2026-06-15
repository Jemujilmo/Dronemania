#!/usr/bin/env python3
"""
record_expert.py — Collect behaviour-cloning demonstrations from a GPS-based
PD scripted expert controller.

The expert mirrors the logic from visualize_race.py (three-zone speed profile,
lookahead yaw blending, IIR-smoothed velocity / altitude targets) but drives
the RacingEnv directly via velocity commands instead of setting motor commands
on the raw simulator.  This guarantees the observations come from the same
20-dim camera obs that RacingEnv(use_camera_obs=True) produces and that the
actions are already in the env's normalised action space ([-1, 1] × 4).

Usage
-----
    python record_expert.py                # 1 000 episodes → records/expert_demos.npz
    python record_expert.py --episodes 500 --max-steps 600
    python record_expert.py --out records/my_demos.npz
"""

import argparse
import sys
import os
import pathlib
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.drone_gym_env import RacingEnv, _VX_MAX, _VY_MAX, _VZ_MAX, _YAW_MAX

# ── Expert hyper-params (tuned to match visualize_race.py behaviour) ──────────
MAX_HORIZONTAL_SPEED = 7.0   # m/s cruise — slightly conservative for reliability
MIN_TURN_SPEED       = 2.5   # m/s floor through tight corners
APPROACH_DIST        = 3.0   # m  — precision zone
DECEL_START_DIST     = 12.0  # m  — start easing speed
VEL_SMOOTH_ALPHA     = 0.15  # IIR gain for velocity target (≈7 Hz at 60 Hz env)
Z_SMOOTH_ALPHA       = 0.10  # IIR gain for altitude target
KP_YAW               = 1.4
KD_YAW               = 0.6
MAX_VZ               = 2.5   # m/s vertical velocity cap


class GpsExpert:
    """
    Stateful expert controller that produces normalised velocity actions for
    RacingEnv given raw simulator access (env._sim).

    State is reset via .reset().  Call .act(env) at every step.
    """

    def __init__(self):
        self.smoothed_vel = np.zeros(2)
        self.smoothed_z   = 1.0

    def reset(self, initial_z: float = 1.0, initial_vxy: np.ndarray = None):
        # Seed the altitude IIR from the actual spawn altitude, not a hardcoded
        # 1.0 default.  RacingEnv spawns at gate-0's altitude which can be 3-5m;
        # starting at 1.0 causes a large spurious descent command on step 0 that
        # would corrupt every episode's first ~30 transitions in the demo buffer.
        self.smoothed_z = initial_z
        # Seed the velocity IIR at the correct initial speed.  Without this,
        # the IIR ramps from [0,0] and the first ~30 steps show vx≈0.08-0.3 —
        # a rare anomalous region in the obs space that BC fails to generalise.
        # Seeding at the proper approach speed makes step-0 indistinguishable
        # from mid-track transitions, fixing the saturated action bug.
        if initial_vxy is not None:
            self.smoothed_vel = initial_vxy.copy()
        else:
            self.smoothed_vel = np.zeros(2)

    def act(self, env: RacingEnv) -> np.ndarray:
        """Return a (4,) normalised action for the current env state."""
        sim   = env._sim
        state = sim.physics.get_state()
        pos   = state['position'].copy()
        vel   = state['velocity'].copy()
        ori   = state['orientation'].copy()
        ang_vel = state['angular_velocity'].copy()
        roll, pitch, yaw = ori

        total_gates = len(sim.gates)
        gi = sim.current_gate_idx

        if gi >= total_gates:
            return np.zeros(4, dtype=np.float32)

        gate_pos = sim.gates[gi].position.copy()

        # ── Approach vector & desired yaw ──────────────────────────────────
        to_gate   = gate_pos - pos
        dist      = float(np.linalg.norm(to_gate))
        to_gate_2d = to_gate[:2]
        dist_2d   = float(np.linalg.norm(to_gate_2d))

        if dist_2d > 0.1:
            desired_yaw = float(np.arctan2(to_gate_2d[1], to_gate_2d[0]))
        else:
            desired_yaw = yaw

        # Lookahead: blend desired_yaw toward exit direction close to gate
        if gi + 1 < total_gates:
            next_gate_pos = sim.gates[gi + 1].position.copy()
            exit_2d = (next_gate_pos - gate_pos)[:2]
            exit_dist = float(np.linalg.norm(exit_2d))
            if exit_dist > 0.1 and dist_2d < 5.0:
                yaw_exit = float(np.arctan2(exit_2d[1], exit_2d[0]))
                blend    = float(np.clip(1.0 - dist_2d / 5.0, 0.0, 1.0))
                diff     = ((yaw_exit - desired_yaw) + np.pi) % (2 * np.pi) - np.pi
                desired_yaw = desired_yaw + blend * diff

        # ── Corner-speed calculation ────────────────────────────────────────
        approach_vec = to_gate_2d / (dist_2d + 1e-6)
        exit_vec     = approach_vec  # default: same direction
        if gi + 1 < total_gates:
            to_next = (sim.gates[gi + 1].position - gate_pos)[:2]
            d_nx    = float(np.linalg.norm(to_next))
            if d_nx > 0.1:
                exit_vec = to_next / d_nx
        cos_turn    = float(np.clip(np.dot(approach_vec, exit_vec), -1.0, 1.0))
        turn_angle  = float(np.arccos(cos_turn))
        corner_factor = float(np.sin(turn_angle))
        cornering_speed = max(MIN_TURN_SPEED,
                              MAX_HORIZONTAL_SPEED * (1.0 - 0.65 * corner_factor))

        # ── Three-zone horizontal speed profile ───────────────────────────
        if dist_2d > 0.1:
            if dist < APPROACH_DIST:
                desired_speed = cornering_speed * (dist / APPROACH_DIST)
            elif dist < DECEL_START_DIST:
                ratio = (dist - APPROACH_DIST) / (DECEL_START_DIST - APPROACH_DIST)
                blend = (1.0 - np.cos(np.pi * ratio)) / 2.0
                desired_speed = cornering_speed + blend * (MAX_HORIZONTAL_SPEED - cornering_speed)
            else:
                desired_speed = MAX_HORIZONTAL_SPEED
            raw_vel_desired = approach_vec * desired_speed
        else:
            raw_vel_desired = np.zeros(2)

        # ── IIR velocity filter ───────────────────────────────────────────
        self.smoothed_vel = (VEL_SMOOTH_ALPHA * raw_vel_desired
                             + (1.0 - VEL_SMOOTH_ALPHA) * self.smoothed_vel)
        desired_vxy = self.smoothed_vel  # world-frame 2D

        # ── Altitude target (IIR-smoothed gate altitude) ──────────────────
        raw_z = float(np.clip(gate_pos[2], 0.5, 7.0))
        self.smoothed_z = Z_SMOOTH_ALPHA * raw_z + (1.0 - Z_SMOOTH_ALPHA) * self.smoothed_z

        alt_error  = self.smoothed_z - pos[2]
        z_vel_cur  = float(vel[2])
        z_command  = float(np.clip(alt_error * 0.8 - z_vel_cur * 0.4, -MAX_VZ, MAX_VZ))

        # Emergency low-altitude override
        if pos[2] < 0.7 or z_vel_cur < -2.5:
            z_command = MAX_VZ
            desired_vxy = vel[:2] * (-0.5)   # gentle braking

        # ── Yaw rate command ──────────────────────────────────────────────
        yaw_error = ((desired_yaw - yaw) + np.pi) % (2 * np.pi) - np.pi
        yaw_rate  = float(np.clip(KP_YAW * yaw_error - KD_YAW * ang_vel[2],
                                  -_YAW_MAX, _YAW_MAX))

        # ── Normalise to [-1, 1] ──────────────────────────────────────────
        action = np.array([
            np.clip(desired_vxy[0] / _VX_MAX, -1.0, 1.0),
            np.clip(desired_vxy[1] / _VY_MAX, -1.0, 1.0),
            np.clip(z_command       / _VZ_MAX, -1.0, 1.0),
            np.clip(yaw_rate        / _YAW_MAX, -1.0, 1.0),
        ], dtype=np.float32)
        return action


# ── Recording loop ─────────────────────────────────────────────────────────────

def record(n_episodes: int = 500, max_steps: int = 650,
           out_path: str = "records/expert_demos.npz",
           verbose: bool = True,
           noise_sigma: float = 0.0,
           explore_prob: float = 0.0) -> None:
    """
    Run the GPS expert for *n_episodes* random tracks (each with a fresh random
    seed) and save all (obs, act) pairs to *out_path*.

    DAgger-style noise injection (noise_sigma > 0):
      With probability *explore_prob* at each step, replace the expert action
      with a clipped-Gaussian-perturbed version before stepping the env.  The
      RECORDED label always remains the GPS expert's ideal action for the
      current state.  This creates training data from off-nominal obs (e.g.
      post-gate overshoot) with correct expert labels, substantially reducing
      covariate shift compared to pure expert imitation.
    """
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    env    = RacingEnv(substeps=4, use_camera_obs=True)
    expert = GpsExpert()

    all_obs  = []
    all_acts = []

    total_steps  = 0
    total_passed = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        initial_z = float(env._sim.physics.get_state()['position'][2])
        # Compute correct initial velocity so the IIR doesn't ramp from zero.
        # This eliminates the startup anomaly in the BC training distribution.
        sim = env._sim
        pos = sim.physics.get_state()['position']
        g0 = sim.gates[0].position
        to_gate = g0[:2] - pos[:2]
        dist = float(np.linalg.norm(to_gate)) + 1e-6
        approach = to_gate / dist
        if len(sim.gates) > 1:
            to_next = (sim.gates[1].position - g0)[:2]
            dn = np.linalg.norm(to_next) + 1e-6
            exit_v = to_next / dn
        else:
            exit_v = approach
        cos_t = float(np.clip(np.dot(approach, exit_v), -1.0, 1.0))
        corner_s = max(MIN_TURN_SPEED, MAX_HORIZONTAL_SPEED * (1.0 - 0.65 * np.sin(np.arccos(cos_t))))
        init_speed = corner_s * min(1.0, dist / APPROACH_DIST)
        initial_vxy = approach * init_speed
        expert.reset(initial_z=initial_z, initial_vxy=initial_vxy)
        ep_obs  = []
        ep_acts = []
        ep_gates = 0

        for _ in range(max_steps):
            expert_action = expert.act(env)

            # DAgger: record ideal expert label, but optionally apply noisy
            # action so the env drifts into off-nominal obs (post-overshoot
            # recovery states, etc.)  The model learns what the expert WOULD
            # do from those recovered/adversarial obs.
            if noise_sigma > 0 and np.random.random() < explore_prob:
                noise = np.random.normal(0, noise_sigma,
                                        size=expert_action.shape)
                apply_action = np.clip(expert_action + noise,
                                      -1.0, 1.0).astype(np.float32)
            else:
                apply_action = expert_action

            ep_obs.append(obs.copy())
            ep_acts.append(expert_action.copy())   # always the IDEAL label

            obs, _reward, terminated, truncated, info = env.step(apply_action)
            ep_gates = info.get('gates_passed', ep_gates)

            if terminated or truncated:
                break

        all_obs.extend(ep_obs)
        all_acts.extend(ep_acts)
        total_steps  += len(ep_obs)
        total_passed += ep_gates

        if verbose and (ep + 1) % max(1, n_episodes // 20) == 0:
            elapsed = time.time() - t0
            avg_g   = total_passed / (ep + 1)
            steps_s = total_steps / max(elapsed, 1e-6)
            print(f"  ep {ep+1:5d}/{n_episodes}  "
                  f"avg_gates={avg_g:.2f}/5  "
                  f"total_steps={total_steps:,}  "
                  f"{steps_s:.0f} steps/s")

    env.close()

    obs_arr = np.array(all_obs,  dtype=np.float32)
    act_arr = np.array(all_acts, dtype=np.float32)

    np.savez_compressed(out, obs=obs_arr, acts=act_arr)

    elapsed = time.time() - t0
    print(f"\nSaved {len(obs_arr):,} transitions to {out}  "
          f"(obs shape {obs_arr.shape}, acts shape {act_arr.shape})  "
          f"in {elapsed:.1f} s")
    print(f"Average gates/episode: {total_passed / n_episodes:.3f} / 5")


def main():
    ap = argparse.ArgumentParser(description="Record expert BC demonstrations")
    ap.add_argument("--episodes",  type=int, default=500,
                    help="Number of episodes to record (default: 500)")
    ap.add_argument("--max-steps", type=int, default=650,
                    help="Max steps per episode (default: 600)")
    ap.add_argument("--out",       default="records/expert_demos.npz",
                    help="Output .npz file (default: records/expert_demos.npz)")
    ap.add_argument("--quiet",     action="store_true",
                    help="Suppress per-episode progress output")
    ap.add_argument("--noise",      type=float, default=0.0,
                    help="Gaussian noise sigma added to applied action (DAgger). "
                         "Recorded label is always the ideal expert action. "
                         "(default: 0.0 = pure imitation)")
    ap.add_argument("--explore-prob", type=float, default=0.30,
                    help="Fraction of steps where noise is applied (default: 0.30)")
    args = ap.parse_args()

    print(f"Recording {args.episodes} episodes × up to {args.max_steps} steps "
          f"→ {args.out}")

    record(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        out_path=args.out,
        verbose=not args.quiet,
        noise_sigma=args.noise,
        explore_prob=args.explore_prob,
    )


if __name__ == "__main__":
    main()
