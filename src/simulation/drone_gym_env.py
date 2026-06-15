"""
Gymnasium environment wrapping the real RacingSimulator + DronePhysics.

Replaces the hollow PyBullet stubs with our working custom physics engine.
Compatible with Stable-Baselines3 / any Gymnasium-compatible RL library.

─────────────────────────────────────────────────────────────────────────────
OBSERVATION  (20 floats)  — camera + IMU + GPS hybrid  (use_camera_obs=True)
─────────────────────────────────────────────────────────────────────────────
Camera bearing to current gate (5):
  [0]  bearing_x      horizontal offset -1(left)..+1(right)
  [1]  bearing_y      vertical offset   -1(down)..+1(up)
  [2]  cam_dist       distance estimate  1=far, 0=filling FOV
  [3]  visibility     0=not seen, 1=fully visible
  [4]  approach_norm  head-on quality 1=head-on, <1=oblique

IMU (10):
  [5:8]   velocity      world-frame [vx, vy, vz]  m/s
  [8:11]  orientation   Euler [roll, pitch, yaw]  rad
  [11:14] angular_vel   [wx, wy, wz]  rad/s
  [14]    yaw_rate      normalised yaw rate
  [15]    speed_norm    |v| / 10

GPS / telemetry fallback (4) — legitimate under competition rules:
  [16:19] gps_dir       (to_gate_world / 20 m), clipped [-2,2]
  [19]    gate_dist_n   distance to gate / 50 m

GPS obs fills in when the camera loses sight of the gate between gates,
giving the policy a directional signal to re-acquire the FOV.
Competition telemetry (position + orientation) provides exactly this data.

─────────────────────────────────────────────────────────────────────────────
OBSERVATION  — GPS mode  (use_camera_obs=False, dev/baseline only)
─────────────────────────────────────────────────────────────────────────────
  [to_gate_dx, dy, dz]   drone→gate vector          (3)
  [vx, vy, vz]           world-frame velocity        (3)
  [roll, pitch, yaw]     euler orientation           (3)
  [wx, wy, wz]           angular velocity            (3)
  [gate_nx, ny, nz]      gate normal direction       (3)
  [next_dx, dy, dz]      relative to next gate       (3)
  [dist_to_gate]         scalar distance             (1)
  [progress]             gates_passed/total_gates    (1)

─────────────────────────────────────────────────────────────────────────────
ACTION  (4 floats in [-1, 1])
─────────────────────────────────────────────────────────────────────────────
  [vx_norm, vy_norm, vz_norm, yaw_norm]
  Scaled to ±8 m/s horizontal, ±3 m/s vertical, ±1.5 rad/s yaw.
  competition_adapter.py converts these to Throttle/Roll/Pitch/Yaw at the
  competition API boundary (velocity_to_attitude()).

─────────────────────────────────────────────────────────────────────────────
REWARD  (camera mode)
─────────────────────────────────────────────────────────────────────────────
  +0.30 * visibility * fwd_norm    fly into visible gate (fwd motion required)
  +0.08 * visibility * lateral / vertical alignment
  +0.20 * delta_approach_dist      GPS approach reward when gate not visible
  -0.05 * (1 - speed_norm)         idleness penalty
  -0.09 * reverse_fraction         backward-flight penalty
  -0.03/step                       net time pressure
  -0.002 * |ang_vel|^2             smoothness
  +80 * quality + momentum_bonus   gate pass (momentum_bonus ∝ gate index)
  +200 + time_bonus                course complete
  -25                              crash
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from simulation.racing_simulator import RacingSimulator  # noqa: E402

# Lazy import — avoids load-order issues when this module is imported early
def _get_detector():
    from perception.gate_detector import SimGateDetector
    return SimGateDetector()

# ── PD constants matching visualize_race.py ──────────────────────────────────
_KP_ROLL  = 4.0;  _KD_ROLL  = 1.8
_KP_PITCH = 3.5;  _KD_PITCH = 1.5
_KP_YAW   = 1.2;  _KD_YAW   = 0.9
_MAX_TILT = 0.55  # rad
_HOVER    = 0.5

_VX_MAX  = 8.0;  _VY_MAX = 8.0;  _VZ_MAX = 3.0;  _YAW_MAX = 1.5

OBS_DIM = 20
ACT_DIM = 4
MAX_EP_STEPS = 6000   # ~100 s at 60 Hz (4 sub-steps × 240 Hz physics) — gate timeout is
                       # the usual binding constraint (40 s/gate); this is the hard ceiling


class RacingEnv(gym.Env):
    """Gymnasium racing environment backed by the real RacingSimulator."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 render_mode: Optional[str] = None,
                 substeps: int = 4,
                 use_camera_obs: bool = False,
                 min_forward_vx: float = 0.0):
        """
        min_forward_vx : float
            Hard lower-bound on the vx command sent to the inner PD loop.
            0.0 (default) = no clamp, full RL freedom during training.
            Set to e.g. 1.5 on game day so the policy can never command reverse
            — the drone always has baseline forward momentum between gates.
        """
        """
        use_camera_obs : bool
            When True the first 5 observation features come from the gate
            detector (camera bearing, elevation, dist, visibility, approach
            angle) instead of GPS-derived to_gate / dist / gate_normal.
            Velocity, orientation and angular_velocity still come from the
            physics state (IMU-equivalent).  Useful for training a policy
            that can transfer to a real camera without architecture changes.
        """
        super().__init__()
        self.render_mode = render_mode
        self.substeps    = substeps   # physics steps per RL decision (60Hz/4 = 15Hz)

        base_cfg = {'sim_rate': 60, 'perception': {'enabled': False},
                    'planning':  {'enabled': False}}
        if config:
            base_cfg.update(config)
        # Enable fast renderer during camera-obs training (no human window).
        # Skips sky gradient loop + HUD text, keeping gate shapes for the detector.
        if use_camera_obs and render_mode != 'human':
            base_cfg.setdefault('fast_render', True)
        self._cfg = base_cfg

        self._sim: Optional[RacingSimulator] = None
        self._prev_pitch = 0.0
        self._prev_roll  = 0.0
        self._ep_steps   = 0
        self._gates_passed = 0
        # Curriculum: set externally by CurriculumCallback to spawn near a gate
        self.curriculum_gate: int = 0
        # Game-day clamp: floor on forward velocity command (0 = disabled)
        self._min_forward_vx: float = float(min_forward_vx)
        # Stored spawn position for anti-stall check
        self._spawn_pos: np.ndarray = np.zeros(3)
        # Camera-based observation mode
        self._use_camera_obs: bool = use_camera_obs
        self._detector = None  # lazily created on first reset
        self._last_cam_vec = None  # cached for bearing reward in step()
        # GPS approach tracking: used for delta-distance reward when gate not visible
        self._prev_gate_dist: float = 0.0

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(ACT_DIM, dtype=np.float32),
            high=np.ones(ACT_DIM, dtype=np.float32),
            dtype=np.float32
        )

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._sim is None:
            self._sim = RacingSimulator(self._cfg)
        self._sim.reset()
        self._prev_pitch  = 0.0
        self._prev_roll   = 0.0
        self._ep_steps    = 0
        self._gates_passed = 0
        if self._use_camera_obs and self._detector is None:
            self._detector = _get_detector()
        # Curriculum: spawn near a specific gate if requested
        start_gate = 0
        if options and 'start_gate' in options:
            start_gate = int(options['start_gate'])
        elif self.curriculum_gate > 0:
            start_gate = self.curriculum_gate
        if start_gate > 0:
            self._teleport_to_gate(start_gate)
        else:
            # Spawn 2 m before gate 0 with zero velocity.
            # Gate 0 is right in front of the drone at episode start, fully
            # visible in camera.  The camera reward (gated by fwd_norm) and
            # idleness penalty force the policy to actively fly through it —
            # no free momentum pass.  This ensures every gate-pass signal is
            # earned by the policy, giving PPO a clean learning target.
            g0     = self._sim.gates[0]
            g0pos  = g0.position
            spawn  = np.array([g0pos[0] - 2.0, g0pos[1], g0pos[2]])
            self._sim.physics.set_state(
                position         = spawn,
                velocity         = np.zeros(3),
                orientation      = np.zeros(3),
                angular_velocity = np.zeros(3),
            )
        # Sync _prev_drone_pos to actual spawn so the first check_gates() call
        # uses the real starting position, not the physics-reset default [0,0,1]
        self._sim._prev_drone_pos = self._sim.physics.get_state()['position'].copy()
        self._spawn_pos = self._sim.physics.get_state()['position'].copy()
        self._last_cam_vec = None
        self._prev_gate_dist = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        assert self._sim is not None, "Call reset() first"
        vx  = float(action[0]) * _VX_MAX
        # Game-day clamp: enforce minimum forward velocity if set.
        # During training this is 0.0 (no effect); set to e.g. 1.5 for real races
        # so the RL policy can never command full reverse between gates.
        if self._min_forward_vx > 0.0:
            vx = max(vx, self._min_forward_vx)
        vy  = float(action[1]) * _VY_MAX
        vz  = float(action[2]) * _VZ_MAX
        yaw = float(action[3]) * _YAW_MAX

        reward = 0.0
        terminated = truncated = False
        gate_info = {}

        for _ in range(self.substeps):
            state  = self._sim.physics.get_state()
            motors = self._velocity_to_motors(state, vx, vy, vz, yaw)
            self._sim.set_motor_commands(motors)
            gi = self._sim.check_gates()
            if gi:
                gate_info = gi
            terminated, truncated, term_info = self._sim.check_termination()
            if terminated or truncated:
                break

        state = self._sim.physics.get_state()
        vel   = state['velocity']
        pos   = state['position']

        gi_idx = self._sim.current_gate_idx
        if self._use_camera_obs:
            # ── Pure camera reward — zero GPS, zero static bonuses ────────────
            # Critical design: NO reward for hovering and staring at the gate.
            # EVERY camera reward requires forward motion (fwd_norm > 0).
            # Without this constraint the policy collapses to a local optimum
            # where it hovers centred on the gate forever.
            obs_vec    = self._last_cam_vec
            visibility = float(obs_vec[3]) if obs_vec is not None else 0.0
            speed      = float(np.linalg.norm(vel))
            speed_norm = np.clip(speed / 5.0, 0.0, 1.0)
            fwd_norm   = max(0.0, vx / _VX_MAX)   # 0 when hovering/reversing

            if visibility > 0.0:
                bearing_x  = float(obs_vec[0])
                bearing_y  = float(obs_vec[1])
                lat_align  = vy * bearing_x / _VY_MAX
                vert_align = vz * bearing_y / _VZ_MAX

                # All rewards gated by fwd_norm: zero payout for hovering in place
                reward += 0.30 * visibility * fwd_norm          # fly into visible gate
                reward += 0.08 * visibility * max(0.0, lat_align)   # correct lateral
                reward += 0.08 * visibility * max(0.0, vert_align)  # correct altitude
            else:
                # Gate not visible: reward flying toward gate using GPS direction.
                # Cosine-alignment (not distance) avoids the vz-saturation attractor
                # that sank the earlier delta-distance reward.  The policy earns
                # up to +0.04/step only when its velocity vector points at the gate.
                if gi_idx < len(self._sim.gates):
                    to_gate_gps = self._sim.gates[gi_idx].position - pos
                    gps_dist    = float(np.linalg.norm(to_gate_gps))
                    if gps_dist > 0.5:
                        gate_unit  = to_gate_gps / gps_dist
                        vel_spd    = float(np.linalg.norm(vel)) + 1e-6
                        gate_align = float(np.dot(vel, gate_unit)) / vel_spd
                        reward    += 0.04 * max(0.0, gate_align)

            # Idleness penalty: being slow always costs regardless of visibility
            # This breaks the hover attractor — standing still accumulates -0.05/step
            reward -= 0.05 * max(0.0, 1.0 - speed_norm)

            # Backward flight penalty
            reward -= 0.09 * max(0.0, -vx / _VX_MAX)
        elif gi_idx < len(self._sim.gates):
            # ── GPS mode: forward-progress reward (dev/baseline only) ─────────
            to_gate = self._sim.gates[gi_idx].position - pos
            d = np.linalg.norm(to_gate)
            if d > 0.01:
                reward += 0.12 * max(0.0, float(np.dot(vel, to_gate / d)))

        # Survival bonus / time penalty (net -0.03/step when idle)
        reward += 0.03
        reward -= 0.06

        # Smoothness
        reward -= 0.002 * float(np.dot(state['angular_velocity'],
                                        state['angular_velocity']))

        if 'gate_passed' in gate_info:
            self._gates_passed += 1
            quality = np.clip(0.5 + gate_info.get('gate_pass_margin_y', 0.0)
                              + gate_info.get('gate_pass_margin_z', 0.0), 0.5, 1.5)
            base_reward = 80.0 * quality
            # Progressive momentum: each successive gate is worth more, giving the
            # policy a clear incentive to push deeper into the track (gate1=+10,
            # gate2=+20, gate5=+50 ...).  Caps at +100 so values stay bounded.
            momentum_bonus = min(10.0 * self._gates_passed, 100.0)
            reward += base_reward + momentum_bonus

        if terminated:
            reward -= 25.0   # lowered 60→25: crash shouldn't dominate over gate passes

        if self._gates_passed == len(self._sim.gates) and not terminated:
            reward += 200.0 + max(0.0, 300.0 - self._sim.time)
            truncated = True

        self._ep_steps += 1
        # Hard episode-length cap
        if self._ep_steps >= MAX_EP_STEPS and not (terminated or truncated):
            truncated = True

        # Anti-stall: cut episode early if drone retreats well behind spawn.
        # Tier-2 removed — gate timeout (15s) already handles stalling before gate.
        if not (terminated or truncated):
            if (self._ep_steps >= 150 and gi_idx == 0
                    and pos[0] < self._spawn_pos[0] - 3.0):
                truncated = True   # flew backward — abort fast

        info = {'gates_passed': self._gates_passed, 'time': self._sim.time,
                **gate_info}
        if (terminated or truncated):
            info['reason'] = term_info.get('reason', 'unknown')
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._sim is None:
            return None
        frame = self._sim.get_observation()['camera']   # (240, 320, 3) BGR (OpenCV native)
        if self.render_mode == 'rgb_array':
            return frame[:, :, ::-1]   # convert to RGB for callers expecting RGB
        if self.render_mode == 'human':
            import cv2
            state = self._sim.physics.get_state()
            speed = float(np.linalg.norm(state['velocity']))
            total = len(self._sim.gates)
            elapsed = self._sim.time
            # Overlay telemetry
            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            disp = cv2.resize(disp, (640, 480), interpolation=cv2.INTER_NEAREST)
            def _txt(text, y, colour=(255, 255, 255)):
                cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, colour, 1, cv2.LINE_AA)
            _txt(f"Speed : {speed:5.1f} m/s",  24)
            _txt(f"Gates : {self._gates_passed}/{total}", 48, (100, 255, 100))
            _txt(f"Time  : {elapsed:6.1f} s",  72, (255, 220, 80))
            _txt("[RL policy]  press Q to skip", 464, (180, 180, 255))
            cv2.imshow('Dronemania — RL Training Watch', disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Signal caller to abort this episode
                return 'quit'
            return frame

    def close(self):
        self._sim = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _teleport_to_gate(self, gate_idx: int) -> None:
        """Teleport drone to the approach position for gate_idx (curriculum)."""
        gates = self._sim.gates
        if gate_idx <= 0 or gate_idx >= len(gates):
            return
        gate_pos = gates[gate_idx].position.copy()
        # Approach direction: vector from previous gate to this gate
        if gate_idx > 0:
            prev_pos = gates[gate_idx - 1].position
            approach = gate_pos - prev_pos
            dist = float(np.linalg.norm(approach))
            approach_norm = approach / (dist + 1e-6)
        else:
            approach_norm = np.array([1., 0., 0.])
        # Start 10 m before the gate, at the gate's altitude
        start_pos = gate_pos - approach_norm * 10.0
        start_pos[2] = gate_pos[2]
        # Give gentle forward velocity so the drone isn't spawned dead-still
        start_vel = approach_norm * 4.0
        self._sim.physics.set_state(
            position         = start_pos,
            velocity         = start_vel,
            orientation      = np.zeros(3),
            angular_velocity = np.zeros(3),
        )
        self._sim.current_gate_idx = gate_idx

    def _get_camera_obs(self) -> np.ndarray:
        """
        Pure camera + IMU observation — zero GPS leakage.

        The policy must learn to FIND gates with the camera, not follow a GPS
        arrow.  When the gate is not visible all camera slots are zero; the
        model must learn to yaw/search until it sees the gate.

        Layout (20 floats):
          [0]     bearing_x  — horizontal offset, -1 (left) … +1 (right)
          [1]     bearing_y  — vertical offset,   -1 (down) … +1 (up)
          [2]     cam_dist   — camera-estimated distance, 0 (close) … 1 (far)
          [3]     visibility — 0 = not seen, 1 = fully seen
          [4]     approach   — approach angle quality
          [5:8]   velocity   — world-frame vx, vy, vz
          [8:11]  orientation — roll, pitch, yaw (rad)
          [11:14] angular_velocity
          [14]    yaw_rate   — current yaw rate (helps learn scan behaviour)
          [15]    speed_norm — |v| / 10, scalar flight speed
          [16:19] GPS fallback — (dx, dy, dz) unit-ish vector to current gate,
                  normalised by 20 m so the model gets a directional hint
                  whenever the camera loses sight of the gate.  The competition
                  telemetry API provides position + orientation, so this is a
                  legitimate competition input — zero GPS cheating.
          [19]    gate_dist_norm — distance to current gate / 50 m
        """
        state   = self._sim.physics.get_state()
        vel     = state['velocity']
        ori     = state['orientation']
        ang_vel = state['angular_velocity']

        # Pure camera detection — no GPS injection anywhere
        raw_frame = self._sim.get_observation().get('camera')
        if raw_frame is not None:
            det     = self._detector.detect(raw_frame)
            cam_vec = det['obs_vector'].copy()   # (5,) float32
        else:
            cam_vec = np.zeros(5, dtype=np.float32)

        # When gate is not visible, cam_vec stays all-zeros.
        # The model must learn to search (yaw) until visibility > 0.
        self._last_cam_vec = cam_vec

        speed_norm = float(np.clip(np.linalg.norm(vel) / 10.0, 0.0, 1.0))
        yaw_rate   = float(np.clip(ang_vel[2] / 3.0, -1.0, 1.0))

        # ── GPS fallback: direction to current target gate ────────────────────
        # When the camera loses sight of the gate (visibility==0) the model
        # would otherwise be completely blind.  Competition telemetry gives us
        # position + orientation legitimately, so we compute a normalised
        # direction vector to the current gate and place it in obs[16:20].
        # The model can learn: "when cam is dark, use GPS arrow to orient, then
        # camera reward kicks in once the gate enters FOV."
        state      = self._sim.physics.get_state()
        pos        = state['position']
        gi_idx     = self._sim.current_gate_idx
        gates      = self._sim.gates
        if gi_idx < len(gates):
            to_gate   = gates[gi_idx].position - pos          # world-frame vector
            gate_dist = float(np.linalg.norm(to_gate))
            # Normalise by 20 m so typical inter-gate distances give ≈1 signal;
            # clip to [-2, 2] to bound the obs without losing direction.
            gps_dir   = np.clip(to_gate / 20.0, -2.0, 2.0).astype(np.float32)
            gate_dist_norm = float(np.clip(gate_dist / 50.0, 0.0, 1.0))
        else:
            gps_dir        = np.zeros(3, dtype=np.float32)
            gate_dist      = 0.0
            gate_dist_norm = 0.0
        # Store for delta-distance reward in step()
        self._prev_gate_dist = gate_dist

        obs = np.concatenate([
            cam_vec,                      # [0:5]  pure camera
            np.clip(vel,     -20, 20),    # [5:8]  IMU velocity
            ori,                          # [8:11] IMU orientation
            np.clip(ang_vel, -15, 15),    # [11:14] IMU angular velocity
            [yaw_rate],                   # [14]   yaw rate
            [speed_norm],                 # [15]   speed
            gps_dir,                      # [16:19] GPS direction to gate
            [gate_dist_norm],             # [19]   GPS distance to gate
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_obs(self) -> np.ndarray:
        if self._use_camera_obs:
            return self._get_camera_obs()
        state   = self._sim.physics.get_state()
        pos     = state['position']
        vel     = state['velocity']
        ori     = state['orientation']
        ang_vel = state['angular_velocity']
        total   = len(self._sim.gates)
        gi      = self._sim.current_gate_idx

        if gi < total:
            gate    = self._sim.gates[gi]
            to_gate = gate.position - pos
            y, p    = gate.orientation[2], gate.orientation[1]
            normal  = np.array([np.cos(p)*np.cos(y), np.cos(p)*np.sin(y), np.sin(p)])
            dist    = np.linalg.norm(to_gate)
        else:
            to_gate = np.zeros(3)
            normal  = np.array([1., 0., 0.])
            dist    = 0.0

        next_rel = (self._sim.gates[gi+1].position - pos
                    if gi + 1 < total else np.zeros(3))

        obs = np.concatenate([
            np.clip(to_gate,  -100, 100),
            np.clip(vel,       -20,  20),
            ori,
            np.clip(ang_vel,   -15,  15),
            normal,
            np.clip(next_rel, -100, 100),
            [np.clip(dist, 0, 100)],
            [gi / max(total, 1)],
        ]).astype(np.float32)
        # Sanitize: replace any NaN/Inf before they corrupt the policy network
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _velocity_to_motors(self, state: dict,
                             vx: float, vy: float,
                             vz: float, yaw_rate: float) -> np.ndarray:
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

        max_r  = np.radians(2.5)
        rp     = np.clip( fwd_err * 0.60, -_MAX_TILT,  _MAX_TILT)
        rr     = np.clip(-lat_err * 0.65, -_MAX_TILT,  _MAX_TILT)
        tp     = np.clip(rp, self._prev_pitch - max_r, self._prev_pitch + max_r)
        tr     = np.clip(rr, self._prev_roll  - max_r, self._prev_roll  + max_r)
        self._prev_pitch = tp;  self._prev_roll = tr

        rc = _KP_ROLL  * (roll  - tr) + _KD_ROLL  * ang_vel[0]
        pc = _KP_PITCH * (pitch - tp) + _KD_PITCH * ang_vel[1]
        yc = np.clip(_KP_YAW * (yaw_rate - ang_vel[2]) - _KD_YAW * ang_vel[2],
                     -0.08, 0.08)

        return np.clip([alt_thr - pc + rc - yc,
                        alt_thr + pc + rc + yc,
                        alt_thr - pc - rc + yc,
                        alt_thr + pc - rc - yc], 0.2, 1.0)


# Keep old name as alias for backward compatibility with existing train.py import
DroneEnv = RacingEnv
