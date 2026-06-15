"""
Lightweight racing drone simulator - custom physics.

Replaces PyBullet with minimal pure-Python implementation.
Includes:
- Gate detection
- Procedural track generation
- Camera simulation
- Performance metrics
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .drone_physics import DronePhysics


@dataclass
class Gate:
    """Gate representation."""
    position: np.ndarray
    orientation: np.ndarray
    width: float  = 1.8   # Wider frame = more pixels at range for vision detection
    height: float = 1.8
    color: Tuple  = (0, 255, 0)  # BGR format (legacy, mostly unused after shape renderer)
    shape: str    = 'square'    # 'square' or 'pentagon'


class RacingSimulator:
    """Lightweight drone racing simulator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize racing simulator."""
        self.config = config
        
        # Physics engine
        self.physics = DronePhysics()
        self.dt = 1.0 / config.get('sim_rate', 240)
        
        # Gates
        self.gates: List[Gate] = []
        self.current_gate_idx = 0
        self.gates_passed_time = {}
        
        # Camera parameters
        self.camera_width  = 640
        self.camera_height = 480
        # 110 deg diagonal FOV — matches typical FPV racing camera.
        # Wide FOV keeps gates in frame longer for visual navigation
        # and provides peripheral visibility for path planning.
        self.fov_degrees = 110

        # fast_render: skip gradient sky, HUD text, gate labels.
        # Enabled automatically during RL training (camera obs without human window)
        # to maximise simulation fps.  The gate shapes are still fully projected so
        # the gate detector receives valid input.
        self.fast_render: bool = bool(config.get('fast_render', False))

        # Pre-build flat sky / ground arrays used by fast render (avoid realloc)
        self._sky_tile   = np.full((120, 320, 3), [187, 221, 235], dtype=np.uint8)
        self._ground_tile= np.full((120, 320, 3), [100, 100,  80], dtype=np.uint8)

        # Print track summary at most once per simulator instance (not every reset)
        self._track_printed = False
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.crashed = False
    
    def reset(self):
        """Reset simulator."""
        self.physics.reset()
        self.time = 0.0
        self.step_count = 0
        self.crashed = False
        self.current_gate_idx = 0
        self.gates_passed_time = {}
        self.last_gate_time = 0.0  # Track time of last gate completion
        self._prev_drone_pos = np.array([0.0, 0.0, 1.0])  # Full 3-D prev position
        
        # Generate procedural track
        self._generate_track()
    
    def _generate_track(self, num_gates: int = 5):
        """Generate Grand Prix race track with full 3D complexity.

        Design goals
        ------------
        X  Monotonic (required for crossed-plane gate detection), but *variable*
           spacing: 8–12 m per sector so gates appear frequently enough for the
           early-stage vision policy to always have a gate nearby.
        Y  Multi-frequency lateral curve that mixes a long sweeper, a medium
           chicane and a high-frequency kink.  Total lateral swing ≈ ±8.3 m.
        Z  Roller-coaster altitude from 0.8 m (ground skim) to 5.5 m (high
           arch), driven by an independent sine so climbs/dives are mostly
           decoupled from the lateral layout.
        """
        self.gates = []
        # Use a fresh random seed every episode so the model learns the visual
        # skill (camera → fly toward gate) rather than memorising one fixed route.
        # For a reproducible eval run, callers can set self._track_seed before reset.
        seed = getattr(self, '_track_seed', None) or np.random.randint(0, 2**31)
        rng = np.random.RandomState(seed)

        # ── Per-episode layout parameters (all driven by rng so every reset is
        #    a genuinely different course, not a memorisable fixed route) ──────────
        # X: per-gate spacing drawn fresh each gate (8–12 m)
        # Y: three sine components with randomised phase offsets and amplitude scales
        y_amp   = [rng.uniform(3.5, 5.5),   # slow sweeper amplitude
                   rng.uniform(2.0, 3.5),   # chicane amplitude
                   rng.uniform(0.6, 1.4)]   # high-freq kink amplitude
        y_phase = [rng.uniform(0, 2 * np.pi),
                   rng.uniform(0, 2 * np.pi),
                   rng.uniform(0, 2 * np.pi)]
        # Z: randomised base height and amplitude (keeps z in [0.8, 5.5])
        z_base  = rng.uniform(2.2, 3.4)
        z_amp   = rng.uniform(1.6, 2.8)
        z_phase = rng.uniform(0, 2 * np.pi)

        # ── Pass 1: build world positions ────────────────────────────────────────
        positions = []
        x = 0.0
        for i in range(num_gates):
            # Variable longitudinal spacing: 8–12 m (randomised per gate)
            x += rng.uniform(8.0, 12.0)

            # Multi-frequency lateral (Y):  sweeper + chicane + kink
            y = (y_amp[0] * np.sin(i * 0.38  + y_phase[0])   # slow sweeper
               + y_amp[1] * np.sin(i * 1.05  + y_phase[1])   # chicane
               + y_amp[2] * np.cos(i * 2.20  + y_phase[2]))  # kink

            # Altitude (Z) roller-coaster: 0.8 m – 5.5 m
            z = z_base + z_amp * np.sin(i * 0.55 + z_phase)
            z = float(np.clip(z, 0.8, 5.5))

            positions.append(np.array([x, y, z], dtype=float))

        # ── Pass 2: orient each gate to face the next one ────────────────────────
        for i, pos in enumerate(positions):
            if i < num_gates - 1:
                delta = positions[i + 1] - pos
            else:
                delta = pos - positions[i - 1]

            yaw   = np.arctan2(delta[1], delta[0])                    # lateral heading
            pitch = np.arctan2(delta[2], np.linalg.norm(delta[:2]))   # climb/dive angle
            roll  = rng.uniform(-0.08, 0.08)                          # slight cant

            # Alternate square / pentagon gates (visually distinct, DRL-style)
            shape = 'pentagon' if (i % 2 == 0) else 'square'
            gate = Gate(
                position=pos,
                orientation=np.array([roll, pitch, yaw]),
                color=(200, 200, 200),  # Neutral white-grey; coloring now comes from shape renderer
                shape=shape,
            )
            self.gates.append(gate)

        # ── Cache envelope for dynamic out-of-bounds checking ────────────────────
        xs = [g.position[0] for g in self.gates]
        ys = [abs(g.position[1]) for g in self.gates]
        self._track_max_x = max(xs) + 30.0
        self._track_max_y = max(ys) + 20.0

        # Print track summary once per simulator instance (silences per-reset spam)
        if not self.fast_render and not self._track_printed:
            self._track_printed = True
            z_lo = min(g.position[2] for g in self.gates)
            z_hi = max(g.position[2] for g in self.gates)
            print(f"[TRACK] {num_gates} gates | "
                  f"X: 0->{max(xs):.0f} m | "
                  f"Y: +/-{max(ys):.1f} m | "
                  f"Z: {z_lo:.1f}-{z_hi:.1f} m")
    
    def set_motor_commands(self, commands: np.ndarray):
        """
        Set motor thrust commands.
        
        Args:
            commands: [m1, m2, m3, m4] in range [0, 1]
        """
        self.physics.step(commands)
        self.step_count += 1
        self.time += self.dt
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current sensor observations."""
        state = self.physics.get_state()
        
        # Render camera image
        camera_image = self._render_camera_view()
        
        # IMU data
        imu_data = {
            'linear_acceleration': state['velocity'] / self.dt,  # Simplified
            'angular_velocity': state['angular_velocity'].copy()
        }
        
        return {
            'camera': camera_image,
            'imu': imu_data,
            'drone_state': state,
            'time': self.time
        }
    
    # ── Gate shape vertex templates (gate-local coords: u=lateral, v=vertical) ──
    # Drone flies through the opening.  Units = fraction of gate half-size,
    # so multiply by (width/2, height/2) to get metres.
    _SQUARE_VERTS = np.array([
        [-1, -1], [ 1, -1], [ 1,  1], [-1,  1]
    ], dtype=float)  # 4 corners, CCW

    # Pentagon = house shape: square base + peaked top.
    # Flying opening is the bottom square portion.
    _PENTAGON_VERTS = np.array([
        [-1,  -1 ],          # 0  bottom-left
        [ 1,  -1 ],          # 1  bottom-right
        [ 1,   0.25],        # 2  right shoulder
        [ 0,   1.0 ],        # 3  peak (centre-top)
        [-1,   0.25],        # 4  left shoulder
    ], dtype=float)  # 5 vertices, CCW

    def _gate_local_basis(self, gate: 'Gate'):
        """Return (right, up) unit vectors in world-space for gate's local frame."""
        roll, pitch, yaw = gate.orientation
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        normal    = np.array([cp * cy, cp * sy, sp])
        gate_right = np.array([-sy,  cy,  0.0])
        gate_up    = np.cross(normal, gate_right)
        n = np.linalg.norm(gate_up)
        if n > 1e-9:
            gate_up /= n
        return gate_right, gate_up

    def _project_world_pt(self, world_pt: np.ndarray,
                          drone_pos, cy, sy, cp, sp, cr, sr,
                          focal: float):
        """
        Project a single world-space 3D point to (px, py, depth) using
        the drone's current yaw/pitch/roll.  Returns None if behind camera.
        """
        rel = world_pt - drone_pos
        # Rotate: -yaw
        x1 =  rel[0]*cy + rel[1]*sy
        y1 = -rel[0]*sy + rel[1]*cy
        z1 =  rel[2]
        # Rotate: -pitch
        x2 = x1*cp - z1*sp
        z2 = x1*sp + z1*cp
        y2 = y1
        # Rotate: -roll
        xb =  x2
        yb =  y2*cr + z2*sr
        zb = -y2*sr + z2*cr
        if xb < 0.1:
            return None
        return (160 + focal * yb / xb,
                120 - focal * zb / xb,
                xb)  # xb == depth

    def _render_camera_view(self) -> np.ndarray:
        """Render camera view: gates drawn as perspective-projected pentagon/square frames."""
        img = np.zeros((240, 320, 3), dtype=np.uint8)

        if self.fast_render:
            # Fast path: two flat blocks — no Python loop, no HUD text
            img[:120] = self._sky_tile
            img[120:] = self._ground_tile
        else:
            # Full quality sky gradient
            for y in range(120):
                img[y, :] = [int(135 + y * 0.5), int(206 + y * 0.3), 235]
            img[120:, :] = [100, 100, 80]

        state     = self.physics.get_state()
        drone_pos = state['position']
        roll, pitch, yaw = state['orientation']

        focal = 320 / np.tan(np.radians(self.fov_degrees / 2))
        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll),  np.sin(roll)

        # Collect visible gates with their depth for painter-sort
        visible = []
        for i, gate in enumerate(self.gates):
            centre_proj = self._project_world_pt(
                gate.position, drone_pos, cy, sy, cp, sp, cr, sr, focal)
            if centre_proj is None:
                continue
            cx_f, cy_f, depth = centre_proj
            # Cull gates whose centre projects far off-screen (not worth projecting)
            if cx_f < -160 or cx_f > 480 or cy_f < -120 or cy_f > 360:
                continue
            visible.append((depth, i, gate, cx_f, cy_f))

        # Painter's algorithm: draw far-to-near
        visible.sort(key=lambda t: t[0], reverse=True)

        for depth, gidx, gate, cx_f, cy_f in visible:
            is_active = (gidx == self.current_gate_idx)
            r_vec, u_vec = self._gate_local_basis(gate)
            hw = gate.width  / 2.0
            hh = gate.height / 2.0

            # Pick vertex template
            tmpl = (self._PENTAGON_VERTS if gate.shape == 'pentagon'
                    else self._SQUARE_VERTS)

            # Project each vertex
            pts_2d = []
            all_visible = True
            for uv in tmpl:
                world_v = gate.position + uv[0] * hw * r_vec + uv[1] * hh * u_vec
                proj = self._project_world_pt(
                    world_v, drone_pos, cy, sy, cp, sp, cr, sr, focal)
                if proj is None:
                    all_visible = False
                    break
                pts_2d.append((int(proj[0]), int(proj[1])))

            if not all_visible or len(pts_2d) < 3:
                continue

            pts_arr = np.array(pts_2d, dtype=np.int32)

            # Thickness: scales with depth (closer = thicker)
            thickness = max(2, int(focal * 0.07 / depth))

            # Colour scheme: all gates use the same white frame — shape is how
            # the detector distinguishes gates.  Active gate gets orange corner
            # accents (small squares at vertices, like DRL bracket LEDs).
            frame_color  = (230, 230, 230)          # off-white pipe
            corner_color = (0, 140, 255) if is_active else (80, 80, 80)  # orange accent if next

            cv2.polylines(img, [pts_arr], isClosed=True,
                          color=frame_color, thickness=thickness)

            # Corner bracket LEDs — drawn AFTER frame so they sit on top
            # Minimum radius 3 px so the orange colour is detectable by HSV mask
            r_dot = max(3, thickness + 1)
            for px, py in pts_2d:
                cv2.circle(img, (px, py), r_dot, corner_color, -1)

            # Gate number label near top vertex (skip in fast_render mode)
            if not self.fast_render:
                top_pt  = min(pts_2d, key=lambda p: p[1])
                lbl_col = (0, 200, 255) if is_active else (160, 160, 160)
                cv2.putText(img, f"G{gidx+1}",
                            (top_pt[0] - 8, max(4, top_pt[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, lbl_col, 1)
        
        # Add HUD info (skip in fast_render mode — detector zeroes those rows anyway)
        if not self.fast_render:
            state = self.physics.get_state()
            pos = state['position']
            vel = np.linalg.norm(state['velocity'])
            
            info_text = [
                f"Pos: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]",
                f"Vel: {vel:.1f} m/s",
                f"Gate: {self.current_gate_idx}/{len(self.gates)} passed",
                f"Time: {self.time:.2f}s"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(img, text, (10, 20 + i*15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def check_gates(self) -> Dict[str, Any]:
        """Check if drone has passed through a gate using oriented gate-plane crossing.

        Gates on the 3-D course have non-zero yaw and pitch, so an X-axis plane
        check fires at the wrong moment for angled gates.  Instead we:
          1. Compute the gate's face-normal in world-space from its yaw+pitch.
          2. Track the signed distance from the drone to that plane each step.
          3. Detect a neg→pos sign change (crossing from back to front).
          4. Interpolate the exact crossing point and project it into gate-local
             lateral / vertical coordinates to decide pass vs miss.
        """
        info = {}
        state = self.physics.get_state()
        drone_pos = state['position']

        if self.current_gate_idx < len(self.gates):
            gate = self.gates[self.current_gate_idx]
            gate_pos  = gate.position
            gate_yaw  = gate.orientation[2]
            gate_pitch = gate.orientation[1]

            # ── Gate basis vectors ──────────────────────────────────────────────
            # Normal: points in the direction the gate faces (forward through the ring)
            cp, sp = np.cos(gate_pitch), np.sin(gate_pitch)
            cy, sy = np.cos(gate_yaw),   np.sin(gate_yaw)
            gate_normal = np.array([cp * cy, cp * sy, sp])

            # Right: perpendicular to normal, lying in the horizontal plane
            gate_right = np.array([-sy, cy, 0.0])

            # Up: perpendicular to both (complete right-hand frame)
            gate_up = np.cross(gate_normal, gate_right)
            norm_up = np.linalg.norm(gate_up)
            if norm_up > 1e-9:
                gate_up /= norm_up

            # ── Signed distances to gate plane ─────────────────────────────────
            d_prev = np.dot(self._prev_drone_pos - gate_pos, gate_normal)
            d_curr = np.dot(drone_pos            - gate_pos, gate_normal)

            # Crossed from back (d<0) to front (d>=0) this step
            if d_prev < 0.0 <= d_curr:
                # Interpolate exact crossing position on the gate plane
                denom = d_curr - d_prev
                t = (-d_prev / denom) if abs(denom) > 1e-9 else 0.5
                crossing = self._prev_drone_pos + t * (drone_pos - self._prev_drone_pos)
                rel = crossing - gate_pos

                # Offsets in gate-local frame
                lateral  = float(np.dot(rel, gate_right))   # +right / −left
                vertical = float(np.dot(rel, gate_up))      # +up   / −down

                half_w = gate.width  / 2  # 0.7 m
                half_h = gate.height / 2  # 0.7 m
                tol = 0.9                  # require within 90 % of half-opening

                info['gate_lateral_offset']  = round(lateral,  3)
                info['gate_vertical_offset'] = round(vertical, 3)

                if abs(lateral) < half_w * tol and abs(vertical) < half_h * tol:
                    self.current_gate_idx += 1
                    info['gate_passed']         = self.current_gate_idx - 1
                    info['gate_pass_margin_y']  = round(half_w * tol - abs(lateral),  3)
                    info['gate_pass_margin_z']  = round(half_h * tol - abs(vertical), 3)
                    self.gates_passed_time[self.current_gate_idx - 1] = self.time
                    self.last_gate_time = self.time
                else:
                    info['gate_missed']  = self.current_gate_idx
                    info['gate_miss_y']  = round(lateral,  2)
                    info['gate_miss_z']  = round(vertical, 2)

        self._prev_drone_pos = drone_pos.copy()
        return info
    
    def check_termination(self) -> Tuple[bool, bool, Dict]:
        """Check if episode should terminate."""
        state = self.physics.get_state()
        drone_pos = state['position']
        
        terminated = False
        truncated = False
        info = {}
        
        # Crash detection (hit ground or too low)
        if drone_pos[2] < 0.1:
            terminated = True
            info['reason'] = 'crashed'
        
        # Course complete
        if self.current_gate_idx >= len(self.gates):
            terminated = True
            info['reason'] = 'course_complete'
            info['total_time'] = self.time
            info['gates_passed'] = len(self.gates)
        
        # Timeout - hard cap (10 min)
        if self.time > 600.0:  # 10 minute hard cap
            truncated = True
            info['reason'] = 'timeout'
        
        # Gate timeout - 15s without progress (was 40s; shorter = bad episodes die
        # faster so the policy gets more high-signal gate-passing experience per phase)
        time_since_last_gate = self.time - self.last_gate_time
        if time_since_last_gate > 15.0:
            truncated = True
            info['reason'] = 'gate_timeout'
            info['time_since_last_gate'] = time_since_last_gate
        
        # Out of bounds - envelope scales with the generated track
        track_max_x = getattr(self, '_track_max_x', 450.0)
        track_max_y = getattr(self, '_track_max_y', 50.0)
        if drone_pos[0] > track_max_x or drone_pos[0] < -10.0:
            terminated = True
            info['reason'] = 'out_of_bounds'
        elif abs(drone_pos[1]) > track_max_y:
            terminated = True
            info['reason'] = 'out_of_bounds'
        elif drone_pos[2] > 18.0:  # Hard ceiling — catches runaway climbs
            terminated = True
            info['reason'] = 'out_of_bounds'
        
        return terminated, truncated, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Step simulation.
        
        Args:
            action: [vx, vy, vz, yaw_rate] velocity commands
            
        Returns:
            (observation, info)
        """
        # Convert velocity commands to motor commands
        # This is a simplified control law
        motor_commands = self._velocity_to_motor_commands(action)
        
        # Step physics
        self.set_motor_commands(motor_commands)
        
        # Get observation
        obs = self.get_observation()
        
        # Check gate passage
        gate_info = self.check_gates()
        
        # Check termination
        terminated, truncated, term_info = self.check_termination()
        
        # Combine info
        info = {**gate_info, **term_info, 'step': self.step_count, 'time': self.time}
        obs['camera'] = obs['camera']  # Return camera image
        
        return obs['camera'], info
    
    def _velocity_to_motor_commands(self, velocity_command: np.ndarray) -> np.ndarray:
        """
        Convert high-level velocity commands to motor thrust values.
        
        This is a simplified control law for demonstration.
        Real implementation would use full cascade control.
        
        Args:
            velocity_command: [vx, vy, vz, yaw_rate]
            
        Returns:
            Motor commands [0-1] for motors 1-4
        """
        vx, vy, vz, yaw_rate = velocity_command
        
        # Hover thrust (half motors)
        hover_thrust = 0.5
        
        # Vertical control (all motors equally)
        # Clamp vz to reasonable range
        vz_clipped = np.clip(vz / 10.0, -0.5, 0.5)  # Normalize to [-0.5, 0.5]
        z_adjust = vz_clipped
        
        # For XY control (simplified quadcopter dynamics)
        # This is very simplified - real control would be more complex
        pitch_command = np.clip(vx / 10.0, -0.3, 0.3)
        roll_command = np.clip(vy / 10.0, -0.3, 0.3)
        
        # Motor commands (in X configuration)
        # m1: front-left, m2: front-right, m3: back-left, m4: back-right
        m1 = hover_thrust + z_adjust + pitch_command - roll_command + yaw_rate * 0.1
        m2 = hover_thrust + z_adjust + pitch_command + roll_command - yaw_rate * 0.1
        m3 = hover_thrust + z_adjust - pitch_command - roll_command - yaw_rate * 0.1
        m4 = hover_thrust + z_adjust - pitch_command + roll_command + yaw_rate * 0.1
        
        motor_commands = np.array([m1, m2, m3, m4])
        
        # Clamp to [0, 1]
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        
        return motor_commands
