"""
competition_adapter.py  —  AI Grand Prix Virtual Qualifier Interface

Bridges the gap between our SB3-PPO model (which outputs velocity setpoints)
and the competition API (which accepts Throttle / Roll / Pitch / Yaw attitude
commands).

─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE RATIONALE
─────────────────────────────────────────────────────────────────────────────
Our training environment uses a hierarchical control stack:

    RL Policy  →  [vx, vy, vz, yaw_rate]  →  PD inner loop  →  motor torques

The competition platform already has its own motor-mixing / inner stabilisation
loop.  It exposes a classical attitude command interface:

    Our Code  →  [Throttle, Roll, Pitch, Yaw]  →  Platform inner loop  →  motors

The conversion is therefore:

    Policy output  →  velocity_to_attitude()  →  competition commands

velocity_to_attitude() is the FIRST HALF of our _velocity_to_motors() function,
stopping before the inner PD that computes per-motor corrections.  We hand the
platform exactly those intermediate attitude setpoints that our PD would have
tried to reach anyway.

─────────────────────────────────────────────────────────────────────────────
COMPETITION TELEMETRY  (legitimately provided by the platform)
─────────────────────────────────────────────────────────────────────────────
  position     np.ndarray (3,)  world-frame metres, [x, y, z]
  velocity     np.ndarray (3,)  world-frame m/s,    [vx, vy, vz]
  orientation  np.ndarray (3,)  Euler rad,           [roll, pitch, yaw]

─────────────────────────────────────────────────────────────────────────────
COMPETITION API OUTPUT
─────────────────────────────────────────────────────────────────────────────
  throttle  float  [0.0 – 1.0]   collective thrust normalised
  roll      float  [rad]         desired roll angle setpoint
  pitch     float  [rad]         desired pitch angle setpoint
  yaw       float  [rad/s]       desired yaw rate

  ⚠ PENDING: exact scale / range will be confirmed once detailed API docs
    arrive.  All conversion constants are grouped in the SCALING section
    below and can be adjusted without touching any other logic.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
    adapter = CompetitionAdapter(
        model_path   = "trained_models/rl_vision_policy_best.zip",
        gate_positions = course.gate_positions,   # list of (x, y, z) tuples
    )

    # On every API tick:
    throttle, roll, pitch, yaw = adapter.step(
        position    = telemetry['position'],
        velocity    = telemetry['velocity'],
        orientation = telemetry['orientation'],
        camera_bgr  = camera_frame,          # H×W×3 uint8 BGR
    )
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import List, Optional, Tuple

# ── Path setup (adapter lives at project root, src/ is one level down) ────────
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Lazy imports so the file loads fast even without GPU / full deps ───────────
def _load_sb3():
    from stable_baselines3 import PPO
    return PPO

def _load_detector():
    from perception.gate_detector import SimGateDetector
    return SimGateDetector()


# ═════════════════════════════════════════════════════════════════════════════
# SCALING CONSTANTS
# These mirror drone_gym_env.py exactly.  Change in both places if you retune.
# ═════════════════════════════════════════════════════════════════════════════
_VX_MAX   = 8.0   # m/s max forward velocity command
_VY_MAX   = 8.0   # m/s max lateral velocity command
_VZ_MAX   = 3.0   # m/s max vertical velocity command
_YAW_MAX  = 1.5   # rad/s max yaw rate command

_HOVER    = 0.5   # baseline throttle to hold altitude
_MAX_TILT = 0.55  # rad maximum roll / pitch angle

# Competition output range (adjust once API docs arrive)
# Throttle: raw internal [0.32, 0.72] remapped to [0.0, 1.0] for the API
_THR_MIN = 0.32
_THR_MAX = 0.72

# GPS obs normalisation (must match drone_gym_env.py)
_GPS_NORM   = 20.0   # metres — normalise to-gate vector by this
_GPS_CLIP   = 2.0    # max clipped value
_DIST_NORM  = 50.0   # metres — normalise gate distance scalar


# ═════════════════════════════════════════════════════════════════════════════
# ATTITUDE CONVERSION
# ═════════════════════════════════════════════════════════════════════════════
def velocity_to_attitude(
    velocity:    np.ndarray,   # world-frame [vx, vy, vz]
    orientation: np.ndarray,   # Euler rad   [roll, pitch, yaw]
    cmd_vx:  float,            # velocity command x  (m/s, scaled)
    cmd_vy:  float,            # velocity command y  (m/s, scaled)
    cmd_vz:  float,            # velocity command z  (m/s, scaled)
    cmd_yaw: float,            # yaw rate command    (rad/s)
) -> Tuple[float, float, float, float]:
    """
    Convert velocity setpoints to attitude commands for the competition API.

    This is the first half of _velocity_to_motors() from drone_gym_env.py,
    stopping BEFORE the inner PD motor-mixing loop that the competition
    platform supplies internally.

    Returns
    -------
    throttle  : float  normalised [0, 1]  — collective thrust
    roll_cmd  : float  radians            — desired roll angle
    pitch_cmd : float  radians            — desired pitch angle
    yaw_cmd   : float  rad/s              — desired yaw rate (pass-through)
    """
    roll, pitch, yaw = float(orientation[0]), float(orientation[1]), float(orientation[2])
    vx_cur, vy_cur, vz_cur = float(velocity[0]), float(velocity[1]), float(velocity[2])

    # --- Throttle (altitude hold + climb rate) --------------------------------
    tilt     = np.sqrt(pitch**2 + roll**2)
    comp     = 1.0 / max(0.87, np.cos(tilt))          # tilt compensation
    vz_err   = cmd_vz - vz_cur
    throttle_raw = np.clip(
        (_HOVER + vz_err * 0.45 - vz_cur * 0.50) * comp,
        _THR_MIN, _THR_MAX
    )
    # Remap raw throttle [0.32, 0.72] → API [0.0, 1.0]
    throttle = (throttle_raw - _THR_MIN) / (_THR_MAX - _THR_MIN)

    # --- Pitch / Roll setpoints (velocity → body-frame tilt targets) ----------
    # Rotate velocity error from world → body frame using current yaw
    vel_err_x = cmd_vx - vx_cur
    vel_err_y = cmd_vy - vy_cur
    cy, sy    = np.cos(yaw), np.sin(yaw)
    fwd_err   =  vel_err_x * cy + vel_err_y * sy   # along body x-axis
    lat_err   = -vel_err_x * sy + vel_err_y * cy   # along body y-axis

    pitch_cmd = float(np.clip( fwd_err * 0.60, -_MAX_TILT, _MAX_TILT))
    roll_cmd  = float(np.clip(-lat_err * 0.65, -_MAX_TILT, _MAX_TILT))

    # --- Yaw: pass through as rate command ------------------------------------
    yaw_cmd   = float(np.clip(cmd_yaw, -_YAW_MAX, _YAW_MAX))

    return throttle, roll_cmd, pitch_cmd, yaw_cmd


# ═════════════════════════════════════════════════════════════════════════════
# GATE TRACKER
# ═════════════════════════════════════════════════════════════════════════════
class GateTracker:
    """
    Tracks which gate the drone is currently targeting based on telemetry.

    The competition provides position telemetry, so we can detect gate passes
    geometrically without needing a separate simulator state variable.
    """
    # Gate pass is triggered when the drone crosses within this radius of
    # the gate centre AND has moved through it (dot product check).
    _PASS_RADIUS = 3.5   # metres — generous to handle timing jitter

    def __init__(self, gate_positions: List[np.ndarray]):
        self.gates      = [np.array(g, dtype=np.float32) for g in gate_positions]
        self.current    = 0
        self._last_pos  = None
        self.n_passed   = 0

    def update(self, position: np.ndarray) -> int:
        """
        Call on every telemetry tick.  Returns current target gate index.
        Advances the index when a gate pass is detected.
        """
        pos = np.asarray(position, dtype=np.float32)

        if self.current >= len(self.gates):
            return self.current   # course complete

        gate_pos = self.gates[self.current]
        dist     = float(np.linalg.norm(pos - gate_pos))

        if dist < self._PASS_RADIUS:
            if self._last_pos is not None:
                # Confirm passage: sign of forward component changed side of gate
                approach = gate_pos - self._last_pos
                through  = pos - gate_pos
                if float(np.dot(approach, through)) > 0:
                    self.current += 1
                    self.n_passed += 1

        self._last_pos = pos.copy()
        return self.current

    def direction_to_gate(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns (unit_vector_to_gate, distance) for the current target gate.
        Used to populate GPS fallback obs[16:20].
        """
        pos = np.asarray(position, dtype=np.float32)
        if self.current >= len(self.gates):
            return np.zeros(3, dtype=np.float32), 0.0
        to_gate = self.gates[self.current] - pos
        dist    = float(np.linalg.norm(to_gate))
        return to_gate, dist


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ADAPTER CLASS
# ═════════════════════════════════════════════════════════════════════════════
class CompetitionAdapter:
    """
    Full inference pipeline: telemetry + camera → competition commands.

    Steps per tick:
      1. Update gate tracker with current telemetry position
      2. Run gate detector on camera frame  →  5-float camera obs vector
      3. Assemble 20-float observation (identical layout to training)
      4. Run SB3 PPO policy  →  [vx_n, vy_n, vz_n, yaw_n] in [-1, 1]
      5. Scale action to physical commands
      6. velocity_to_attitude()  →  (throttle, roll, pitch, yaw)
      7. Return 4 competition commands
    """

    def __init__(
        self,
        model_path:     str,
        gate_positions: List,
        min_forward_vx: float = 1.5,   # m/s  — clamp: no pure reverse on game day
    ):
        """
        Parameters
        ----------
        model_path       : path to .zip file from stable-baselines3 PPO.save()
        gate_positions   : list of (x, y, z) tuples / arrays for all course gates
        min_forward_vx   : minimum forward velocity command (prevents pure reverse).
                           Set to 0.0 to disable (training default).
        """
        # Load model
        PPO = _load_sb3()
        self.model          = PPO.load(model_path, device='cpu')
        self._detector      = _load_detector()
        self._gate_tracker  = GateTracker(gate_positions)
        self._min_fwd_vx    = float(min_forward_vx)
        self._prev_gate_dist: float = 0.0

        print(f"[CompetitionAdapter] loaded model from {model_path}")
        print(f"[CompetitionAdapter] course has {len(gate_positions)} gates")
        print(f"[CompetitionAdapter] min_forward_vx = {min_forward_vx} m/s")

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        position:    np.ndarray,   # [x, y, z]  metres
        velocity:    np.ndarray,   # [vx, vy, vz]  m/s
        orientation: np.ndarray,   # [roll, pitch, yaw]  radians
        angular_velocity: np.ndarray = None,  # [wx, wy, wz]  rad/s — if available
        camera_bgr:  np.ndarray = None,       # H×W×3 uint8 BGR camera frame
    ) -> Tuple[float, float, float, float]:
        """
        Run one inference step.

        Returns
        -------
        (throttle, roll_cmd, pitch_cmd, yaw_cmd)
          throttle  : [0.0 – 1.0]   collective thrust
          roll_cmd  : radians        desired roll angle
          pitch_cmd : radians        desired pitch angle
          yaw_cmd   : rad/s          desired yaw rate
        """
        position    = np.asarray(position,    dtype=np.float32)
        velocity    = np.asarray(velocity,    dtype=np.float32)
        orientation = np.asarray(orientation, dtype=np.float32)
        ang_vel     = (np.asarray(angular_velocity, dtype=np.float32)
                       if angular_velocity is not None
                       else np.zeros(3, dtype=np.float32))

        # 1. Update gate tracker
        self._gate_tracker.update(position)

        # 2. Camera detection
        if camera_bgr is not None:
            det     = self._detector.detect(camera_bgr)
            cam_vec = det['obs_vector'].copy()   # (5,) float32
        else:
            cam_vec = np.zeros(5, dtype=np.float32)

        # 3. GPS fallback: direction to current target gate
        to_gate, gate_dist = self._gate_tracker.direction_to_gate(position)
        gps_dir      = np.clip(to_gate / _GPS_NORM, -_GPS_CLIP, _GPS_CLIP).astype(np.float32)
        gate_dist_n  = float(np.clip(gate_dist / _DIST_NORM, 0.0, 1.0))

        # 4. Assemble observation — MUST match drone_gym_env._get_camera_obs() exactly
        speed_norm = float(np.clip(np.linalg.norm(velocity) / 10.0, 0.0, 1.0))
        yaw_rate   = float(np.clip(ang_vel[2] / 3.0, -1.0, 1.0))

        obs = np.concatenate([
            cam_vec,                          # [0:5]   camera bearing
            np.clip(velocity,    -20, 20),    # [5:8]   IMU velocity
            orientation,                      # [8:11]  IMU orientation
            np.clip(ang_vel,     -15, 15),    # [11:14] IMU angular velocity
            [yaw_rate],                       # [14]    yaw rate
            [speed_norm],                     # [15]    speed norm
            gps_dir,                          # [16:19] GPS direction to gate
            [gate_dist_n],                    # [19]    GPS distance to gate
        ]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # 5. Policy inference
        action, _ = self.model.predict(obs, deterministic=True)

        # 6. Scale action [-1, 1] → physical commands
        vx  = float(action[0]) * _VX_MAX
        vy  = float(action[1]) * _VY_MAX
        vz  = float(action[2]) * _VZ_MAX
        yaw = float(action[3]) * _YAW_MAX

        # Game-day floor: never command full reverse between gates
        if self._min_fwd_vx > 0.0:
            vx = max(vx, self._min_fwd_vx)

        # 7. Convert to attitude commands for competition API
        throttle, roll_cmd, pitch_cmd, yaw_cmd = velocity_to_attitude(
            velocity    = velocity,
            orientation = orientation,
            cmd_vx  = vx,
            cmd_vy  = vy,
            cmd_vz  = vz,
            cmd_yaw = yaw,
        )

        return throttle, roll_cmd, pitch_cmd, yaw_cmd

    @property
    def gates_passed(self) -> int:
        """Number of gates cleared so far this run."""
        return self._gate_tracker.n_passed

    @property
    def current_gate(self) -> int:
        """Index of the gate currently being targeted (0-based)."""
        return self._gate_tracker.current

    def reset(self, gate_positions: Optional[List] = None):
        """
        Call between laps / retries.  Resets gate tracker.
        Pass gate_positions to update course layout (useful if it changes run-to-run).
        """
        if gate_positions is not None:
            self._gate_tracker = GateTracker(gate_positions)
        else:
            self._gate_tracker.current   = 0
            self._gate_tracker.n_passed  = 0
            self._gate_tracker._last_pos = None
        self._prev_gate_dist = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE INTEGRATION SKELETON (to be filled once API docs arrive)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    """
    Skeleton showing how competition_adapter.py slots into the DCL Python API.

    Replace the placeholder calls (api.get_telemetry(), api.get_camera(), etc.)
    with the actual interface once DCL's detailed docs arrive.

    Expected flow:
      1. api.connect() / api.load_course()
      2. Create CompetitionAdapter with gate_positions from course definition
      3. api.start_run()  →  event loop
      4. On each tick: get telemetry + camera → adapter.step() → api.send_commands()
      5. Repeat until lap complete
    """
    import time

    MODEL_PATH = 'trained_models/rl_vision_policy_best.zip'

    # ── Placeholder: get course gate positions from competition API ────────────
    # When the API is available this will be something like:
    #   gate_positions = api.get_course().gate_positions
    # For local testing, use our sim gate positions:
    gate_positions = [
        [10.0,  0.0, 2.5],
        [22.0,  4.0, 3.0],
        [36.0, -3.0, 2.0],
        [50.0,  5.0, 3.5],
        [65.0,  0.0, 2.5],
    ]

    adapter = CompetitionAdapter(
        model_path      = MODEL_PATH,
        gate_positions  = gate_positions,
        min_forward_vx  = 1.5,
    )

    print("\n[Skeleton] Ready.  Waiting for competition API docs to hook up.")
    print("[Skeleton] Output format: (throttle, roll_rad, pitch_rad, yaw_rad_s)")

    # ── Fake one tick to verify the pipeline runs end-to-end ──────────────────
    fake_position    = np.array([8.0, 0.0, 2.5], dtype=np.float32)
    fake_velocity    = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    fake_orientation = np.array([0.0, 0.05, 0.0], dtype=np.float32)
    fake_camera      = np.zeros((240, 320, 3), dtype=np.uint8)   # black frame

    try:
        result = adapter.step(
            position         = fake_position,
            velocity         = fake_velocity,
            orientation      = fake_orientation,
            angular_velocity = np.zeros(3),
            camera_bgr       = fake_camera,
        )
        throttle, roll_cmd, pitch_cmd, yaw_cmd = result
        print(f"\n[Skeleton] Test tick output:")
        print(f"  Throttle : {throttle:.4f}  (0=min, 1=max)")
        print(f"  Roll     : {np.degrees(roll_cmd):+.2f} deg")
        print(f"  Pitch    : {np.degrees(pitch_cmd):+.2f} deg")
        print(f"  Yaw rate : {np.degrees(yaw_cmd):+.2f} deg/s")
        print(f"  Gate     : {adapter.current_gate} / {len(gate_positions)}")
        print("\n[Skeleton] Pipeline OK.  Ready to connect to competition API.")
    except Exception as e:
        print(f"\n[Skeleton] Pipeline test failed: {e}")
        print("  Is the model file present at the path above?")
