#!/usr/bin/env python3
"""
Real-time racing AI visualization.

Shows:
- Camera view with detected gates
- Drone trajectory in 3D
- Gate positions
- Telemetry (position, velocity, altitude)
"""

import sys
import os
import time
import json
import pathlib
import datetime
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_simulator import RacingSimulator
from perception.gate_detector import GateDetector
from planning.waypoint_planner import WaypointPlanner
from control.simple_pid_controller import SimplePIDController


RECORDS_FILE = pathlib.Path(__file__).parent / "records" / "race_records.json"


def load_records() -> dict:
    """Load best-time records from disk. Returns empty record if file absent."""
    if RECORDS_FILE.exists():
        try:
            with open(RECORDS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'best_time': None, 'best_gates': 0, 'best_date': None,
            'attempts': 0, 'best_splits': [], 'history': []}


def save_records(records: dict) -> None:
    """Persist race records to disk."""
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)


def draw_record_hud(img: np.ndarray, elapsed: float, gates_passed: int,
                   total_gates: int, records: dict, gate_splits: list) -> np.ndarray:
    """Top-right HUD: best time, current delta, attempt counter."""
    h, w = img.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    best  = records.get('best_time')

    lines = []
    if best is not None:
        lines.append((f"BEST  {best:.2f}s", (215, 180, 0)))   # gold
        delta = elapsed - best
        d_col = (0, 200, 0) if delta < 0 else (0, 80, 255)
        lines.append((f"  \u0394  {delta:+.2f}s", d_col))
        # Per-gate delta: compare current split to best split
        gi = len(gate_splits)
        if gi > 0 and gi <= len(records.get('best_splits', [])):
            seg_delta = gate_splits[-1] - records['best_splits'][gi - 1]
            s_col = (0, 200, 0) if seg_delta < 0 else (0, 80, 255)
            lines.append((f"G{gi:02d}  {seg_delta:+.2f}s", s_col))
    else:
        lines.append(("BEST  --.-s", (160, 160, 160)))

    attempts = records.get('attempts', 0)
    lines.append((f"RUN #{attempts + 1}", (160, 160, 160)))

    # Draw background box
    box_w, box_h = 185, len(lines) * 26 + 12
    x0 = w - box_w - 6
    y0 = 150
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (100, 100, 100), 1)

    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (x0 + 8, y0 + 22 + i * 26), font, 0.55, col, 1)

    return img


def draw_telemetry(img, drone_state, target, gate_num, total_gates, elapsed_time):
    """Draw telemetry overlay on image."""
    h, w = img.shape[:2]

    # Semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (w-5, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Text data
    pos = drone_state['position']
    vel = drone_state['velocity']
    vel_mag = np.linalg.norm(vel)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Time: {elapsed_time:.1f}s", (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Gate: {gate_num}/{total_gates}", (10, 50), font, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Pos: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]", (10, 75), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Vel: {vel_mag:.1f} m/s", (10, 100), font, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"Alt: {pos[2]:.2f}m", (10, 125), font, 0.6, (255, 0, 255), 2)

    # Target indicator
    if target is not None:
        target_pos = target.get('position', [0, 0, 0])
        cv2.putText(img, f"Target: [{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}]",
                    (10, h - 20), font, 0.4, (100, 200, 255), 1)

    return img


def draw_detection_overlay(img, perception_result, drone_state=None, target=None, scale=3):
    """
    Rich real-time detection overlay.

    Draws:
      - Scaled bounding ellipses for every detected gate (next gate highlighted)
      - Crosshair reticle + dashed line toward the next-gate target
      - Velocity vector arrow (lateral + vertical components)
      - Detection mask minimap (top-right corner)
      - Speed-zone badge (Cruise / Decel / Precision)
    """
    h, w = img.shape[:2]

    # ── 1. Detection bounding boxes ──────────────────────────────────────────
    detected = perception_result.get('gates', [])
    next_gate = perception_result.get('next_gate')

    for i, gate in enumerate(detected[:6]):
        bx, by, bw, bh = [v * scale for v in gate['bbox']]
        cx, cy = gate['center'][0] * scale, gate['center'][1] * scale
        score = gate.get('score', 0)
        conf  = gate.get('confidence', 0)

        is_next = (next_gate is not None and gate is next_gate)
        box_color = (0, 255, 0) if is_next else (0, 200, 255)   # green / orange

        # Bounding rectangle (dashed for non-next gates)
        if is_next:
            cv2.rectangle(img, (int(bx), int(by)), (int(bx+bw), int(by+bh)), box_color, 2)
        else:
            # Dashed by drawing short segments
            dash = 12
            for d in range(0, int(bw), dash * 2):
                cv2.line(img, (int(bx+d), int(by)), (int(min(bx+d+dash, bx+bw)), int(by)), box_color, 1)
                cv2.line(img, (int(bx+d), int(by+bh)), (int(min(bx+d+dash, bx+bw)), int(by+bh)), box_color, 1)
            for d in range(0, int(bh), dash * 2):
                cv2.line(img, (int(bx), int(by+d)), (int(bx), int(min(by+d+dash, by+bh))), box_color, 1)
                cv2.line(img, (int(bx+bw), int(by+d)), (int(bx+bw), int(min(by+d+dash, by+bh))), box_color, 1)

        # Center dot
        cv2.circle(img, (int(cx), int(cy)), 4 if is_next else 2, box_color, -1)

        # Score label
        label = f"S:{score:.0f} C:{conf:.2f}" if is_next else f"S:{score:.0f}"
        cv2.putText(img, label, (int(bx), int(by) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

    # ── 2. Target reticle + path line toward next gate ───────────────────────
    if next_gate is not None:
        tx = int(next_gate['center'][0] * scale)
        ty = int(next_gate['center'][1] * scale)
        r  = 18

        # Crosshair arms
        cv2.line(img, (tx - r, ty), (tx + r, ty), (0, 255, 0), 2)
        cv2.line(img, (tx, ty - r), (tx, ty + r), (0, 255, 0), 2)
        # Circle reticle
        cv2.circle(img, (tx, ty), r, (0, 255, 0), 2)
        # Dashed line from image centre to target
        sx0, sy0 = w // 2, h // 2
        dist2 = np.hypot(tx - sx0, ty - sy0)
        if dist2 > 1:
            steps = int(dist2 / 20)
            for s in range(steps):
                alpha = s / max(steps, 1)
                beta  = (s + 0.5) / max(steps, 1)
                p1 = (int(sx0 + alpha * (tx - sx0)), int(sy0 + alpha * (ty - sy0)))
                p2 = (int(sx0 + beta  * (tx - sx0)), int(sy0 + beta  * (ty - sy0)))
                cv2.line(img, p1, p2, (0, 200, 0), 1)

    # ── 3. Velocity arrow ─────────────────────────────────────────────────────
    if drone_state is not None:
        vel = drone_state['velocity']
        # In camera space: vy → image X (right), -vz → image Y (up)
        arrow_scale = 22
        ax = int(w // 2 + vel[1] * arrow_scale)
        ay = int(h // 2 - vel[2] * arrow_scale)
        speed = np.linalg.norm(vel[:2])
        # Colour: green when slow, red when fast
        r_c = int(min(255, speed / 8.0 * 255))
        g_c = int(max(0, 255 - speed / 8.0 * 255))
        vel_color = (0, g_c, r_c)
        cv2.arrowedLine(img, (w // 2, h // 2), (ax, ay), vel_color, 2, tipLength=0.25)
        cv2.circle(img, (w // 2, h // 2), 5, (255, 255, 255), -1)

    # ── 4. Detection mask minimap (top-right corner) ─────────────────────────
    mask = perception_result.get('detection_mask')
    if mask is not None:
        pw, ph = 160, 120   # minimap size
        mini = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
        # Colour the mask green-on-black for clarity
        mini_bgr = np.zeros((ph, pw, 3), dtype=np.uint8)
        mini_bgr[mini > 0] = (0, 220, 0)
        # Overlay bounding boxes on minimap (scaled from 320x240 → 160x120)
        ms = 0.5
        for gate in detected[:6]:
            bx, by, bw2, bh2 = [int(v * ms) for v in gate['bbox']]
            is_n = (next_gate is not None and gate is next_gate)
            mc = (0, 255, 0) if is_n else (0, 180, 255)
            cv2.rectangle(mini_bgr, (bx, by), (bx+bw2, by+bh2), mc, 1)
        # Border
        cv2.rectangle(mini_bgr, (0, 0), (pw-1, ph-1), (180, 180, 180), 1)
        cv2.putText(mini_bgr, "DETECT MASK", (4, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        # Blend onto top-right
        x0, y0 = w - pw - 5, 5
        roi = img[y0:y0+ph, x0:x0+pw]
        cv2.addWeighted(mini_bgr, 0.85, roi, 0.15, 0, roi)
        img[y0:y0+ph, x0:x0+pw] = roi

    # ── 5. Speed-zone badge ───────────────────────────────────────────────────
    if target is not None:
        gate_dist = target.get('gate_distance', float('inf'))
        if gate_dist < 3.0:
            zone_label, zone_color = "PRECISION", (0, 80, 255)
        elif gate_dist < 12.0:
            zone_label, zone_color = "DECEL", (0, 200, 255)
        else:
            zone_label, zone_color = "CRUISE", (0, 255, 100)
        badge_x = w - 175
        cv2.rectangle(img, (badge_x, h - 32), (w - 5, h - 5), (30, 30, 30), -1)
        cv2.rectangle(img, (badge_x, h - 32), (w - 5, h - 5), zone_color, 1)
        cv2.putText(img, f"{zone_label}  {gate_dist:.1f}m", (badge_x + 6, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)

    return img


def draw_gate_flash(img: np.ndarray, flash: dict) -> np.ndarray:
    """Render a gate-pass or gate-miss flash overlay and decrement its timer."""
    if flash['timer'] <= 0:
        return img

    h, w = img.shape[:2]
    alpha = min(1.0, flash['timer'] / 15.0)  # fade out over last 15 frames
    color = flash['color']

    # Coloured border pulse
    bw = max(4, int(10 * alpha))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, bw)

    # Central badge background
    font     = cv2.FONT_HERSHEY_DUPLEX
    main_txt = flash['text']
    sub_txt  = flash['subtext']
    main_sz  = cv2.getTextSize(main_txt, font, 1.3, 3)[0]
    bx = (w - main_sz[0]) // 2
    by = h // 2 - 10
    pad = 12
    box_x0 = bx - pad
    box_y0 = by - main_sz[1] - pad
    box_x1 = bx + main_sz[0] + pad
    box_y1 = by + pad
    if sub_txt:
        sub_sz  = cv2.getTextSize(sub_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        box_y1 += sub_sz[1] + pad + 6
    cv2.rectangle(img, (box_x0, box_y0), (box_x1, box_y1), (10, 10, 10), -1)
    cv2.rectangle(img, (box_x0, box_y0), (box_x1, box_y1), color, 2)

    # Main label
    cv2.putText(img, main_txt, (bx, by), font, 1.3, color, 3)

    # Sub-label (crossing accuracy)
    if sub_txt:
        sub_sz = cv2.getTextSize(sub_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        sx = (w - sub_sz[0]) // 2
        cv2.putText(img, sub_txt, (sx, by + sub_sz[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    flash['timer'] -= 1
    return img


def draw_cockpit_targeting(img, drone_state, gate, gate_idx, total_gates,
                           fov_degrees=110):
    """
    Cockpit targeting overlay that shows the actual gate opening the drone must
    thread, plus live lateral/vertical offset gauges.

    Draws (all in 960x720 upscaled coordinates):
      - The projected 1.8x1.8m passage quad  (green = aligned, red = off)
      - Exact gate-center crosshair
      - Dashed aim-line from camera-center to gate
      - Bottom HUD: lateral bar  |  vertical bar  |  ALIGNED / MISALIGNED badge
    """
    if gate is None:
        return img

    h, w   = img.shape[:2]
    SCALE  = 3               # render scale: camera (320x240) -> screen (960x720)
    FOV    = fov_degrees     # must match simulator
    IMG_W, IMG_H = 320, 240
    focal  = IMG_W / np.tan(np.radians(FOV / 2))   # ~554 px

    drone_pos = drone_state['position']
    drone_yaw = drone_state['orientation'][2]

    # ── Gate basis vectors (same as check_gates) ────────────────────────────
    yaw   = gate.orientation[2]
    pitch = gate.orientation[1]
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    gate_normal = np.array([cp*cy, cp*sy, sp])
    gate_right  = np.array([-sy,   cy,   0.0])
    gup_raw     = np.cross(gate_normal, gate_right)
    gup_len     = np.linalg.norm(gup_raw)
    gate_up     = gup_raw / gup_len if gup_len > 1e-9 else np.array([0., 0., 1.])

    half_w = gate.width  / 2   # 0.7 m
    half_h = gate.height / 2   # 0.7 m
    tol    = 0.9               # matches check_gates threshold

    # ── Live drone offset in gate-local frame ────────────────────────────────
    rel      = drone_pos - gate.position
    lateral  = float(np.dot(rel, gate_right))
    vertical = float(np.dot(rel, gate_up))
    distance = float(np.linalg.norm(gate.position - drone_pos))

    inside_lat  = abs(lateral)  < half_w * tol
    inside_vert = abs(vertical) < half_h * tol
    aligned     = inside_lat and inside_vert

    # ── Projection helper (320x240 camera -> 960x720 screen) ─────────────────
    drone_pitch = drone_state['orientation'][1]
    drone_roll  = drone_state['orientation'][0]
    cp, sp = np.cos(drone_pitch), np.sin(drone_pitch)
    cr, sr = np.cos(drone_roll),  np.sin(drone_roll)

    def project(world_pt):
        r = world_pt - drone_pos
        cos_y, sin_y = np.cos(drone_yaw), np.sin(drone_yaw)
        # Step 1: yaw  (Rz^T)
        x1 =  r[0]*cos_y + r[1]*sin_y
        y1 = -r[0]*sin_y + r[1]*cos_y
        z1 =  r[2]
        # Step 2: pitch  (Ry^T)  — positive pitch lifts nose, shifts gate upward in frame
        x2 = x1*cp - z1*sp
        z2 = x1*sp + z1*cp
        y2 = y1
        # Step 3: roll  (Rx^T)
        xb =  x2
        yb =  y2*cr + z2*sr
        zb = -y2*sr + z2*cr
        if xb < 0.1:
            return None
        px = IMG_W/2 + focal * yb / xb
        py = IMG_H/2 - focal * zb / xb
        return (int(round(px * SCALE)), int(round(py * SCALE)))

    gate_ctr_screen = project(gate.position)

    # 4 corners of the passage opening in world space
    corners_w = [
        gate.position + gate_right*(-half_w) + gate_up* half_h,   # TL
        gate.position + gate_right* half_w   + gate_up* half_h,   # TR
        gate.position + gate_right* half_w   + gate_up*(-half_h), # BR
        gate.position + gate_right*(-half_w) + gate_up*(-half_h), # BL
    ]
    corners_s = [project(c) for c in corners_w]
    gate_visible = gate_ctr_screen is not None and all(c is not None for c in corners_s)

    if gate_visible:
        pts = np.array(corners_s, dtype=np.int32)
        cx_s, cy_s = gate_ctr_screen

        # Choose color by alignment state
        if distance > 25:
            fill_c, frame_c = (120, 120, 0), (180, 180, 40)    # distant - dim cyan
        elif aligned:
            fill_c, frame_c = (0, 130, 0),  (0, 255, 60)       # green - on target
        else:
            fill_c, frame_c = (0, 30, 160), (0, 70, 255)       # red - off target

        # Semi-transparent passage zone fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], fill_c)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

        # Passage frame outline
        cv2.polylines(img, [pts], True, frame_c, 2)

        # Corner anchor dots
        for pt in corners_s:
            cv2.circle(img, pt, 4, frame_c, -1)

        # Gate-center precision crosshair
        arm = 14
        cv2.line(img, (cx_s-arm, cy_s), (cx_s+arm, cy_s), frame_c, 2)
        cv2.line(img, (cx_s, cy_s-arm), (cx_s, cy_s+arm), frame_c, 2)
        cv2.circle(img, (cx_s, cy_s), 4, frame_c, -1)

        # Dashed aim-line: camera center -> gate center
        sx0, sy0 = w//2, h//2
        dist_px = np.hypot(cx_s-sx0, cy_s-sy0)
        if dist_px > 20:
            steps = max(3, int(dist_px / 22))
            for s in range(steps):
                a = s / steps;  b = (s + 0.45) / steps
                p1 = (int(sx0 + a*(cx_s-sx0)), int(sy0 + a*(cy_s-sy0)))
                p2 = (int(sx0 + b*(cx_s-sx0)), int(sy0 + b*(cy_s-sy0)))
                cv2.line(img, p1, p2, frame_c, 1)

        # Distance + gate number label above the quad
        top_y = max(min(pts[:,1]) - 18, 14)
        lbl   = f"G{gate_idx+1}  {distance:.1f}m"
        lw    = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        lx    = int(np.clip(cx_s - lw//2, 4, w-lw-4))
        cv2.putText(img, lbl, (lx, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_c, 1)

    # ── Drone camera-center dot (the drone is always here) ───────────────────
    cv2.circle(img, (w//2, h//2), 7, (255, 255, 255), 2)
    cv2.drawMarker(img, (w//2, h//2), (200, 200, 200),
                   cv2.MARKER_CROSS, 20, 1)

    # ── Bottom HUD gauges ────────────────────────────────────────────────────
    # Lateral bar  (left half of bottom strip)
    BAR_Y  = h - 50
    BAR_H  = 18
    L_X0   = 14
    L_LEN  = 250
    L_X1   = L_X0 + L_LEN
    L_MID  = L_X0 + L_LEN // 2

    def _draw_offset_bar(x0, x1, mid, bar_y, bar_h, value, half_range, label_str, inside):
        # Background
        cv2.rectangle(img, (x0, bar_y), (x1, bar_y+bar_h), (25, 25, 25), -1)
        cv2.rectangle(img, (x0, bar_y), (x1, bar_y+bar_h), (90, 90, 90), 1)
        # Safe zone (green band)
        safe_px = int((L_LEN/2) * tol)
        cv2.rectangle(img, (mid-safe_px, bar_y+2), (mid+safe_px, bar_y+bar_h-2), (0, 70, 0), -1)
        # Gate-edge ticks at ±half_range
        edge_px = int((x1-x0)/2 * (half_range*tol) / (half_range*2))
        cv2.line(img, (mid-edge_px, bar_y), (mid-edge_px, bar_y+bar_h), (100,180,100), 1)
        cv2.line(img, (mid+edge_px, bar_y), (mid+edge_px, bar_y+bar_h), (100,180,100), 1)
        # Centre tick
        cv2.line(img, (mid, bar_y), (mid, bar_y+bar_h), (180,180,180), 1)
        # Indicator needle
        needle_x = int(np.clip(value/(half_range*2), -0.5, 0.5) * (x1-x0) + mid)
        needle_c  = (0, 210, 0) if inside else (0, 50, 255)
        cv2.line(img, (needle_x, bar_y-3), (needle_x, bar_y+bar_h+3), needle_c, 3)
        # Label
        cv2.putText(img, label_str, (x0, bar_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, needle_c, 1)

    _draw_offset_bar(L_X0, L_X1, L_MID, BAR_Y, BAR_H,
                     lateral,  half_w, f"LAT  {lateral:+.2f}m",  inside_lat)

    V_X0  = L_X1 + 18
    V_LEN = 250
    V_X1  = V_X0 + V_LEN
    V_MID = V_X0 + V_LEN // 2
    _draw_offset_bar(V_X0, V_X1, V_MID, BAR_Y, BAR_H,
                     vertical, half_h, f"VERT {vertical:+.2f}m", inside_vert)

    # Alignment status badge
    status_txt   = "ALIGNED" if aligned else "OFF-TARGET"
    status_color = (0, 210, 0) if aligned else (0, 50, 255)
    cv2.putText(img, status_txt,
                (V_X1 + 14, BAR_Y + BAR_H),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)

    return img


def velocity_to_motors(velocity_cmd: np.ndarray) -> np.ndarray:
    """Convert velocity commands to motor thrusts with smooth control."""
    vx, vy, vz = velocity_cmd
    
    # Emergency: if any component is too extreme, stabilize
    if abs(vx) > 8.0 or abs(vy) > 8.0 or abs(vz) > 5.0:
        return np.array([0.5, 0.5, 0.5, 0.5])  # Emergency hover
    
    hover_base = 0.5
    
    # Altitude control - responsive but smooth
    z_adj = np.clip(vz * 0.08, -0.18, 0.18)
    z_throttle = np.clip(hover_base + z_adj, 0.35, 0.65)
    
    # Pitch/roll - conservative for stability
    pitch_cmd = np.clip(vx / 15.0 * 0.12, -0.12, 0.12)
    roll_cmd = np.clip(vy / 15.0 * 0.12, -0.12, 0.12)
    
    # Apply motor mixing
    m1 = z_throttle + pitch_cmd - roll_cmd
    m2 = z_throttle + pitch_cmd + roll_cmd
    m3 = z_throttle - pitch_cmd - roll_cmd
    m4 = z_throttle - pitch_cmd + roll_cmd
    
    motors = np.array([m1, m2, m3, m4])
    motors = np.clip(motors, 0.3, 0.7)
    
    return motors


def main():
    """Run visualized racing AI."""
    
    config = {
        'sim_rate': 60,  # Reduced from 240 for 4× faster simulation
        'perception': {
            'enabled': True,
            'camera': {
                'input_resolution': [640, 480],
                'processing_resolution': [320, 240],
            },
            'gate_detection': {
                'method': 'color_threshold',
                'min_confidence': 0.1,
                'color_ranges': {
                    'lower': [10, 20, 20],
                    'upper': [180, 255, 255],
                },
            },
        },
        'planning': {
            'enabled': True,
            'method': 'reactive',
            'lookahead_distance': 12.0,  # Long lookahead so intermediate target stays far ahead
            'target_velocity': 8.0,
            'camera_center_bias_px': 10.0,  # Tuned: 0px=0.18m left, 25px=0.49m right, 10px≈centered
            'collision_check': True,
            'replan_rate_hz': 20,
        },
    }
    
    # Initialize
    sim = RacingSimulator(config)
    detector = GateDetector(config['perception'])
    planner = WaypointPlanner(config['planning'])
    
    sim.reset()
    
    print("=" * 70)
    print("RACING AI - LIVE VISUALIZATION")
    print("=" * 70)
    print(f"\nTrack: {len(sim.gates)} gates")
    print("Press 'q' to quit, SPACE to pause\n")
    
    race_start = time.time()
    step = 0
    paused = False
    gates_passed = 0
    gate_splits   = []                        # elapsed time at each gate pass
    gate_flash = {'timer': 0, 'color': (0, 255, 0), 'text': '', 'subtext': ''}

    # Load time records from disk
    records = load_records()
    records['attempts'] = records.get('attempts', 0) + 1

    # Persistent state for motion smoothing (prevents snapping on gate transitions)
    smoothed_vel_desired = np.zeros(2)   # IIR-filtered horizontal velocity target
    smoothed_z_target    = 1.0           # IIR-filtered altitude target
    Z_SMOOTH_ALPHA       = 0.08          # ~5Hz bandwidth — smooth climbs, no altitude jerks
    prev_target_pitch    = 0.0           # For tilt rate limiting
    prev_target_roll     = 0.0
    VEL_SMOOTH_ALPHA     = 0.12          # Tighter: absorbs direction changes more (~6Hz at 60Hz)
    MAX_TILT_RATE_RAD    = np.radians(2.5)  # Max tilt change per step (2.5°/step = 150°/s)
    
    # Create window
    cv2.namedWindow('Racing AI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Racing AI', 960, 720)
    
    try:
        while True:
            if not paused:
                # Get observation
                obs = sim.get_observation()
                camera_image = obs['camera']
                drone_state = obs['drone_state']
                
                # Gate detection
                perception = detector.process(camera_image)
                
                # Add current gate index and actual gate position to help planner filter
                perception['current_gate_idx'] = sim.current_gate_idx
                perception['total_gates'] = len(sim.gates)
                if sim.current_gate_idx < len(sim.gates):
                    perception['expected_gate_pos'] = sim.gates[sim.current_gate_idx].position
                
                # Planning (camera-based navigation)
                target = planner.plan(
                    current_state=drone_state,
                    perception=perception
                )
                
                # Debug: extensive diagnosis (reduced frequency for speed)
                if step % 100 == 0:
                    drone_pos = drone_state['position']
                    drone_yaw = drone_state['orientation'][2]
                    
                    # Show current gate we should be targeting
                    next_idx = sim.current_gate_idx
                    print(f"\n[GATE {next_idx+1}] Drone:[{drone_pos[0]:.1f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}]", end="")
                    
                    if next_idx < len(sim.gates):
                        target_gate = sim.gates[next_idx]
                        print(f" -> Target Gate {next_idx+1}:[{target_gate.position[0]:.1f}, {target_gate.position[1]:.2f}]", end="")
                    
                    # Show what detector found
                    gates_detected = perception.get('gates', [])
                    if gates_detected:
                        next_gate = perception.get('next_gate')
                        if next_gate:
                            score = next_gate.get('score', 0)
                            print(f" | Detected:{next_gate['center']} Area:{next_gate['area']:.0f} Score:{score:.1f}", end="")
                    
                    print(f" | Planner:[{target['position'][0]:.1f}, {target['position'][1]:.2f}]")
                
                
                # Get current drone state
                drone_pos = drone_state['position']
                drone_vel = drone_state['velocity']
                
                # Use planner's Z target — IIR-smoothed to eliminate step jumps
                # between lookahead waypoints and suppress tilt-coupling spikes.
                raw_z = np.clip(target['position'][2], 0.5, 7.0)
                smoothed_z_target = Z_SMOOTH_ALPHA * raw_z + (1.0 - Z_SMOOTH_ALPHA) * smoothed_z_target
                planner_z = smoothed_z_target
                target_pos = np.array([target['position'][0],
                                      target['position'][1],
                                      planner_z])
                target_altitude = planner_z
                position_error = target_pos - drone_pos
                
                # Emergency altitude protection: trigger early and strong
                current_altitude = drone_pos[2]
                z_vel_current = drone_vel[2]
                
                if current_altitude < 0.7 or z_vel_current < -3.0:
                    # EMERGENCY: Critically low altitude or rapid descent only
                    z_command = 3.0  # Maximum climb command
                    xy_velocity = -drone_vel[:2] * 0.8  # Strong braking
                else:
                    # Normal altitude control with damping (PD controller)
                    z_error = position_error[2]
                    z_command = np.clip(z_error * 0.8 - z_vel_current * 0.3, -2.0, 2.0)
                    
                    # Normal horizontal control
                    xy_error = position_error[:2]
                    xy_vel_current = drone_vel[:2]
                    xy_mag = np.linalg.norm(xy_error)
                    
                    if xy_mag > 0.1:
                        xy_direction = xy_error / xy_mag
                        max_horiz_vel = 2.5
                        xy_vel_desired = xy_direction * min(max_horiz_vel, xy_mag * 0.6)
                        xy_velocity = xy_vel_desired - xy_vel_current * 0.4
                    else:
                        xy_velocity = -xy_vel_current * 0.6
                
                desired_vel = np.concatenate([xy_velocity, [z_command]])
                
                # CONSERVATIVE SLOW-SPEED CONTROLLER
                # Focus: Gate navigation accuracy over speed
                
                # Get current state
                euler = drone_state['orientation']  # [roll, pitch, yaw]
                ang_vel = drone_state.get('angular_velocity', np.zeros(3))  # [wx, wy, wz]
                roll, pitch, yaw = euler
                xy_vel = drone_vel[:2]
                xy_speed = np.linalg.norm(xy_vel)
                
                # Altitude control with smooth PD  – no deadband, tilt-compensation
                altitude_error = target_pos[2] - current_altitude
                altitude_error = np.clip(altitude_error, -2.0, 2.0)

                # PD: Kp raised so that 1m error → +0.45 throttle above hover.
                # Kd scaled to kill vertical oscillation (critically damped at ~0.5 band).
                altitude_command = altitude_error * 0.45 - z_vel_current * 0.50

                # Tilt-compensation: forward/lateral pitch steals vertical lift (cos rule).
                # Cap at 30° (cos≈0.87) to avoid overcorrecting on small steering inputs,
                # which themselves cause the up/down pulses we're trying to remove.
                tilt_mag = np.sqrt(pitch**2 + roll**2)
                lift_comp = 1.0 / max(0.87, np.cos(tilt_mag))  # cap at ~30° (1/0.87≈1.15)

                altitude_throttle = np.clip((0.5 + altitude_command) * lift_comp, 0.32, 0.72)
                
                # Safety: if drone is significantly above the current gate target, gently reduce
                # throttle (soft penalty, not hard clamp – avoids the sudden snap)
                gate_z_target = (sim.gates[sim.current_gate_idx].position[2]
                                 if sim.current_gate_idx < len(sim.gates) else 3.0)
                overshoot = current_altitude - (gate_z_target + 3.5)
                if overshoot > 0:
                    altitude_throttle = max(0.32, altitude_throttle - np.clip(overshoot * 0.05, 0.0, 0.12))
                
                # Tuned PD controller for attitude - maximum racing aggression
                Kp_roll = 4.0   # Maximum responsiveness
                Kd_roll = 1.8   # Very strong damping
                Kp_pitch = 3.5  # Very aggressive forward control  
                Kd_pitch = 1.5
                
                # VELOCITY CONTROL - three-phase: cruise → smooth corner decel → precision
                MAX_HORIZONTAL_SPEED = 8.0   # m/s open-track cruise
                MIN_TURN_SPEED      = 2.5    # m/s floor through tight turns
                APPROACH_DIST       = 3.0    # m  - precision zone (very close to gate)
                DECEL_START_DIST    = 12.0   # m  - begin deceleration for upcoming corner

                xy_error = target_pos[:2] - drone_pos[:2]
                xy_distance = np.linalg.norm(xy_error)
                gate_dist = target.get('gate_distance', xy_distance)  # actual distance to gate

                # --- Corner-speed calculation + desired yaw ----------------------------
                # Measure the heading change required at the next gate so we commit to
                # a speed that lets the drone complete the turn without overshooting.
                gate_idx = sim.current_gate_idx
                approach_vec = np.array([1.0, 0.0])   # default: straight forward
                exit_vec     = np.array([1.0, 0.0])
                desired_yaw  = yaw   # fallback: maintain current heading
                if gate_idx < len(sim.gates):
                    gate_pos_2d = sim.gates[gate_idx].position[:2]
                    to_gate = gate_pos_2d - drone_pos[:2]
                    d_to = np.linalg.norm(to_gate)
                    if d_to > 0.1:
                        approach_vec = to_gate / d_to
                        desired_yaw  = np.arctan2(to_gate[1], to_gate[0])
                    # Exit vector: current gate → next gate
                    if gate_idx + 1 < len(sim.gates):
                        next_gpos = sim.gates[gate_idx + 1].position[:2]
                        to_next = next_gpos - gate_pos_2d
                        d_nx = np.linalg.norm(to_next)
                        if d_nx > 0.1:
                            exit_vec = to_next / d_nx
                            # Blend desired yaw toward exit direction only very close to gate
                            # (5 m) so the drone is pointing out the far side at crossing.
                            if 0.1 < d_to < 5.0:
                                yaw_exit = np.arctan2(to_next[1], to_next[0])
                                blend = np.clip(1.0 - d_to / 5.0, 0.0, 1.0)
                                diff = ((yaw_exit - desired_yaw) + np.pi) % (2*np.pi) - np.pi
                                desired_yaw = desired_yaw + blend * diff

                cos_turn = np.clip(np.dot(approach_vec, exit_vec), -1.0, 1.0)
                turn_angle = np.arccos(cos_turn)          # 0 = straight, π/2 = 90° turn
                corner_factor = np.sin(turn_angle)        # 0 straight → 1 at 90°
                # Speed target at the gate: full speed when straight, slowed for turns
                cornering_speed = max(MIN_TURN_SPEED,
                                      MAX_HORIZONTAL_SPEED * (1.0 - 0.65 * corner_factor))

                # --- Three-zone speed profile ----------------------------------------
                if xy_distance > 0.1:
                    xy_direction = xy_error / xy_distance
                    lateral_error = abs(xy_error[1])  # Y-axis offset from gate line

                    if gate_dist < APPROACH_DIST:
                        # Zone 3 – Precision: linear ramp cornering_speed → 0 across APPROACH_DIST.
                        # Continuous with Zone 2 at the boundary (gate_dist == APPROACH_DIST → cornering_speed).
                        desired_speed = cornering_speed * (gate_dist / APPROACH_DIST)

                    elif gate_dist < DECEL_START_DIST:
                        # Zone 2 – Smooth decel: cosine ease-in from cruise → cornering_speed.
                        # ratio=1 at DECEL_START_DIST boundary (cruise), ratio=0 at APPROACH_DIST boundary.
                        ratio = (gate_dist - APPROACH_DIST) / (DECEL_START_DIST - APPROACH_DIST)
                        blend = (1.0 - np.cos(np.pi * ratio)) / 2.0   # 0 near gate, 1 far away
                        desired_speed = cornering_speed + blend * (MAX_HORIZONTAL_SPEED - cornering_speed)
                        # Gentle lateral penalty keeps drone tracking the gate line (eased to /8)
                        desired_speed *= np.exp(-lateral_error / 8.0)

                    else:
                        # Zone 1 – Cruise: full speed, no lateral penalty (tilt handles steering)
                        desired_speed = MAX_HORIZONTAL_SPEED

                    raw_vel_desired = xy_direction * desired_speed
                else:
                    raw_vel_desired = np.zeros(2)

                # IIR low-pass filter on velocity target to absorb gate-transition jumps
                # and any other sudden direction changes.
                smoothed_vel_desired = (VEL_SMOOTH_ALPHA * raw_vel_desired
                                        + (1.0 - VEL_SMOOTH_ALPHA) * smoothed_vel_desired)
                xy_vel_desired = smoothed_vel_desired
                
                # Velocity error (desired - actual)
                xy_vel_error = xy_vel_desired - xy_vel[:2]
                
                # Lateral correction computed in body frame below (before tilt calc)
                LATERAL_CORRECTION_GAIN = 0.50  # kept for reference but no longer applied directly
                
                # ── Yaw PD  ──────────────────────────────────────────────────────────────
                # Physics: torque_yaw = (m2+m3 − m1−m4)*0.02  → positive cmd = CCW (+yaw)
                # Motor mix below: CCW motors (m2,m3) get +yaw_correction → correct sign.
                Kp_yaw      = 1.2   # conservative — prevents oscillation
                Kd_yaw      = 0.9   # heavy damping
                yaw_error   = ((desired_yaw - yaw) + np.pi) % (2*np.pi) - np.pi
                yaw_correction = np.clip(
                    Kp_yaw * yaw_error - Kd_yaw * ang_vel[2], -0.08, 0.08)

                # ── Velocity→tilt in BODY frame ──────────────────────────────────────────
                # Project world-frame velocity error onto drone heading so that
                # pitch = acceleration in the direction we're facing, roll = lateral.
                # NOTE: no separate position correction — it fights the velocity term
                # during turns (body-frame lateral pos error flips sign mid-yaw).
                cos_y2 = np.cos(yaw);  sin_y2 = np.sin(yaw)
                body_fwd_err   =  xy_vel_error[0]*cos_y2 + xy_vel_error[1]*sin_y2
                body_right_err = -xy_vel_error[0]*sin_y2 + xy_vel_error[1]*cos_y2

                # Tilt commands (velocity error only — no double-correction position term)
                VEL_TO_TILT_FORWARD = 0.60   # rad per (m/s)
                VEL_TO_TILT_LATERAL = 0.65   # slightly higher than forward but not 1.0

                raw_pitch = np.clip( body_fwd_err   * VEL_TO_TILT_FORWARD, -0.6,  0.6)
                raw_roll  = np.clip(-body_right_err * VEL_TO_TILT_LATERAL, -0.55, 0.55)

                # Emergency brake only if way over speed
                if xy_speed > MAX_HORIZONTAL_SPEED * 1.5:
                    raw_pitch = 0.0
                    raw_roll  = 0.0

                # Rate-limit tilt to prevent snap on gate transitions / sudden target jumps
                target_pitch = np.clip(raw_pitch,
                                       prev_target_pitch - MAX_TILT_RATE_RAD,
                                       prev_target_pitch + MAX_TILT_RATE_RAD)
                target_roll  = np.clip(raw_roll,
                                       prev_target_roll  - MAX_TILT_RATE_RAD,
                                       prev_target_roll  + MAX_TILT_RATE_RAD)
                prev_target_pitch = target_pitch
                prev_target_roll  = target_roll
                
                # Stabilize toward target angles
                roll_correction = (Kp_roll * (roll - target_roll) + Kd_roll * ang_vel[0])
                pitch_correction = (Kp_pitch * (pitch - target_pitch) + Kd_pitch * ang_vel[1])
                
                # Motor mixing - DERIVED FROM PHYSICS ENGINE TORQUE EQUATIONS
                # Physics: torque_roll  = (m3+m4 − m1−m2)*0.1
                #          torque_pitch = (m1+m3 − m2−m4)*0.1
                #          torque_yaw   = (m2+m3 − m1−m4)*0.02   (m1,m4=CW  m2,m3=CCW)
                # Goal: Create torque = -correction (to oppose the error)
                # Yaw:  positive yaw_correction → increase CCW pair (m2,m3) → -(CW pair m1,m4)
                m1 = altitude_throttle - pitch_correction + roll_correction - yaw_correction  # Front-right (CW)
                m2 = altitude_throttle + pitch_correction + roll_correction + yaw_correction  # Front-left  (CCW)
                m3 = altitude_throttle - pitch_correction - roll_correction + yaw_correction  # Back-left   (CCW)
                m4 = altitude_throttle + pitch_correction - roll_correction - yaw_correction  # Back-right  (CW)
                
                motors = np.clip([m1, m2, m3, m4], 0.2, 1.0)
                
                # Diagnostic: only print on genuine crash-risk altitude (<0.45 m)
                if current_altitude < 0.45 and step % 50 == 0:
                    print(f"[EMERGENCY LOW ALT] Alt:{current_altitude:.2f}m Vz:{z_vel_current:.2f}m/s")
                
                # Alert on safety activations (only when the control actually fires)
                if current_altitude > gate_z_target + 3.5:
                    print(f"[ALTITUDE LIMIT] Forcing descent - Alt:{current_altitude:.2f}m (gate_z={gate_z_target:.1f}m)")
                
                # Debug: show detailed velocity vectors (reduced frequency)
                if step % 100 == 0 and not paused:
                    print(f"  Vel:[{xy_vel[0]:.2f},{xy_vel[1]:.2f}] VErr:[{xy_vel_error[0]:.2f},{xy_vel_error[1]:.2f}] Tilt R:{np.degrees(target_roll):.1f}° P:{np.degrees(target_pitch):.1f}°")

                
                sim.set_motor_commands(motors)
                
                # Check gates
                gate_info = sim.check_gates()
                if 'gate_passed' in gate_info:
                    gates_passed += 1
                    lat   = gate_info.get('gate_lateral_offset', 0.0)
                    vert  = gate_info.get('gate_vertical_offset', 0.0)
                    mar_y = gate_info.get('gate_pass_margin_y', 0.0)
                    mar_z = gate_info.get('gate_pass_margin_z', 0.0)
                    print(f"  [PASSED] Gate {gates_passed}/5 @ {time.time() - race_start:.2f}s  "
                          f"lat:{lat:+.3f}m vert:{vert:+.3f}m  margin Y:{mar_y:.3f}m Z:{mar_z:.3f}m")
                    gate_splits.append(time.time() - race_start)  # record split
                    gate_flash['timer']   = 70
                    gate_flash['color']   = (0, 220, 0)         # green
                    gate_flash['text']    = f"GATE {gates_passed} CLEAR"
                    gate_flash['subtext'] = f"lat:{lat:+.2f}m  vert:{vert:+.2f}m  margin Y:{mar_y:.2f}m Z:{mar_z:.2f}m"
                    # Seed IIR with current lateral velocity so the vel-error on the
                    # first frame after a gate pass is near zero (no deceleration spike)
                    smoothed_vel_desired = drone_vel[:2].copy()
                    smoothed_z_target    = drone_pos[2]  # re-seed altitude filter to current alt
                    prev_target_pitch = 0.0
                    prev_target_roll  = 0.0
                elif 'gate_missed' in gate_info:
                    g  = gate_info['gate_missed']
                    dy = gate_info['gate_miss_y']
                    dz = gate_info['gate_miss_z']
                    print(f"  [MISSED] Gate {g+1} -- lat:{dy:+.2f}m  vert:{dz:+.2f}m")
                    gate_flash['timer']   = 100
                    gate_flash['color']   = (0, 60, 255)        # red
                    gate_flash['text']    = f"GATE {g+1} MISSED"
                    gate_flash['subtext'] = f"lat:{dy:+.2f}m  vert:{dz:+.2f}m"
                
                # Check termination
                terminated, truncated, term_info = sim.check_termination()
                
                if terminated and not paused:
                    # Print debug info at crash
                    print(f"\n[DEBUG] Crash details:")
                    print(f"  Reason: {term_info.get('reason')}")
                    print(f"  Position: {drone_pos}")
                    print(f"  Velocity: {drone_vel}, mag={np.linalg.norm(drone_vel):.2f}")
                    print(f"  Target: {target_pos}")
                    print(f"  Gates detected: {len(perception.get('gates', []))}")
                
                step += 1
            
            # Visualization (camera_image is already BGR natively from the simulator)
            vis_img = camera_image.copy()
            
            # Upscale for better visibility
            vis_img = cv2.resize(vis_img, (960, 720), interpolation=cv2.INTER_NEAREST)
            
            # Draw gate detections + full detection overlay
            vis_img = draw_detection_overlay(vis_img, perception, drone_state, target, scale=3)

            # Cockpit targeting: projected gate opening + live offset gauges
            target_gate = (sim.gates[sim.current_gate_idx]
                           if sim.current_gate_idx < len(sim.gates) else None)
            vis_img = draw_cockpit_targeting(vis_img, drone_state, target_gate,
                                             sim.current_gate_idx, len(sim.gates),
                                             fov_degrees=sim.fov_degrees)

            # Gate pass/miss flash overlay
            vis_img = draw_gate_flash(vis_img, gate_flash)

            # Record HUD (best time + delta + run number)
            elapsed = time.time() - race_start
            vis_img = draw_record_hud(vis_img, elapsed, gates_passed,
                                      len(sim.gates), records, gate_splits)

            # Draw telemetry
            elapsed = time.time() - race_start
            vis_img = draw_telemetry(vis_img, drone_state, target, gates_passed, 5, elapsed)
            
            # Show status
            if paused:
                cv2.putText(vis_img, "PAUSED", (vis_img.shape[1]//2 - 100, vis_img.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            # Display
            cv2.imshow('Racing AI', vis_img)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[QUIT] User stopped visualization")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'[PAUSED]' if paused else '[RESUMED]'}")
            
            # Termination (check both terminated and truncated)
            if not paused and (terminated or truncated):
                reason = term_info.get('reason', 'unknown')
                elapsed_final = time.time() - race_start
                is_clean = gates_passed == len(sim.gates)

                print(f"\nFinal stats:")
                print(f"  Gates passed: {gates_passed}/{len(sim.gates)}")
                print(f"  Time: {elapsed_final:.2f}s  |  Steps: {step}")

                if is_clean:
                    prev_best = records.get('best_time')
                    if prev_best is None or elapsed_final < prev_best:
                        records['best_time']   = round(elapsed_final, 3)
                        records['best_gates']  = gates_passed
                        records['best_date']   = datetime.date.today().isoformat()
                        records['best_splits'] = gate_splits[:]
                        label = "NEW RECORD" if prev_best else "FIRST RECORD"
                        print(f"  [{label}] {elapsed_final:.2f}s" +
                              (f"  (was {prev_best:.2f}s)" if prev_best else ""))
                    else:
                        print(f"  vs Best: {records['best_time']:.2f}s  (Δ{elapsed_final - records['best_time']:+.2f}s)")
                    records.setdefault('history', []).append({
                        'time': round(elapsed_final, 3),
                        'gates': gates_passed,
                        'date': datetime.date.today().isoformat(),
                        'splits': gate_splits[:]
                    })
                    records['history'] = records['history'][-50:]
                    save_records(records)
                    print(f"  [SAVED] records/race_records.json")
                else:
                    print(f"  Reason: {reason}")

                # Overlay final result for 3 s
                color = (0, 220, 0) if is_clean else (0, 60, 255)
                msg   = (f"COMPLETE  {elapsed_final:.2f}s" if is_clean
                         else f"ENDED: {reason.upper()}")
                cv2.putText(vis_img, msg,
                            (vis_img.shape[1]//2 - 280, vis_img.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.imshow('Racing AI', vis_img)
                cv2.waitKey(3000)
                break
            
            # Minimal delay for responsive visualization (faster testing)
            if not paused:
                time.sleep(0.0001)  # Very small delay - run as fast as possible
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User stopped race")
    finally:
        cv2.destroyAllWindows()
    
    print(f"\nFinal stats:")
    print(f"  Gates passed: {gates_passed}/5")
    print(f"  Time: {time.time() - race_start:.2f}s")
    print(f"  Steps: {step}")


if __name__ == "__main__":
    main()
