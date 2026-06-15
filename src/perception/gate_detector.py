"""
Gate Detector - perception module for Dronemania / AI-GP.

Detects racing gates rendered as perspective-projected pentagon or square frames.
Uses edge detection + polygon contour fitting (no colour dependency).

DRL gate shapes rendered by racing_simulator.py:
  Pentagon  - 5 vertices, house shape (alternating even-numbered gates)
  Square    - 4 vertices, rectangular frame (odd-numbered gates)

All gate frames are drawn as off-white (230,230,230) pipe outlines.
Active (next) gate has orange corner LED dots - used as secondary cue only.

Detection pipeline
------------------
1. Greyscale + Gaussian blur  (suppresses JPEG noise and sky gradient bands)
2. Canny edge detection
3. findContours (external only)
4. approxPolyDP  ->  4-point or 5-point polygon candidates
5. Filter by area, convexity, aspect-ratio plausibility
6. Separate orange-dot scan to tag the active gate
7. Score by area * centring; pick best candidate
8. Return rich dict + 5-element obs_vector for RL

Observation vector (5 floats, approx [-1,1]):
  [0] bearing_norm    (cx - 160) / 160
  [1] elevation_norm  (120 - cy) / 120
  [2] dist_norm       1 - min(area/REF_AREA, 1)   1=far  0=filling fov
  [3] visibility      0.3-1.0 scaled by area if active (orange) gate; 0.0 if wrong gate; 0.15 if orange-only (no frame)
  [4] approach_norm   min(w,h) / max(w,h)          1=head-on  <1=oblique

Usage
-----
    from src.perception.gate_detector import SimGateDetector
    det = SimGateDetector()
    result = det.detect(bgr_frame)   # bgr_frame: H x W x 3 uint8 BGR
    obs_vec = result['obs_vector']   # np.ndarray (5,) float32
    vis     = det.draw_overlay(bgr_frame, result)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMG_W, _IMG_H = 320, 240
_CX,    _CY    = _IMG_W // 2, _IMG_H // 2

# Area (px**2) treated as "gate filling most of FOV" for dist_norm normalisation.
# A 1.4 m gate at 4 m produces roughly 70x70 px raw -> ~3600 px**2 contour area.
_REF_AREA = 4_000.0

_MIN_AREA = 35     # smaller blobs are noise / distant labels
_MAX_AREA_RATIO = 0.7   # reject if contour fills >70% of image (background bleed)

# Orange corner LED colour range (BGR -> HSV) on the ACTIVE gate.
# (0, 140, 255) BGR  ->  H~16, S~255, V~255  in OpenCV HSV
_ORANGE_LO = np.array([10, 180, 180], dtype=np.uint8)
_ORANGE_HI = np.array([22, 255, 255], dtype=np.uint8)

# Polygon approximation epsilon as fraction of contour arc length
_APPROX_EPS = 0.04

# Accepted vertex counts for recognised shapes.
# 4-5 = crisp polygon at mid-range.
# 6-9 = rounded corners from edge dilation at close range.
_VALID_VERTS = {4, 5, 6, 7, 8, 9}


# ---------------------------------------------------------------------------
# Internal candidate class
# ---------------------------------------------------------------------------
class _Candidate:
    __slots__ = ('cx', 'cy', 'w', 'h', 'area', 'n_verts', 'is_active', 'score', 'poly')

    def __init__(self, cx, cy, w, h, area, n_verts, poly):
        self.cx       = cx
        self.cy       = cy
        self.w        = w
        self.h        = h
        self.area     = area
        self.n_verts  = n_verts
        self.is_active = False
        self.score    = 0.0
        self.poly     = poly   # (N,2) int32 vertices


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class SimGateDetector:
    """
    Shape-based gate detector for pentagon and square frame gates.
    Works on BGR frames produced by RacingSimulator._render_camera_view().
    """

    def __init__(self,
                 min_area:      int   = _MIN_AREA,
                 ref_area:      float = _REF_AREA,
                 canny_lo:      int   = 40,
                 canny_hi:      int   = 120,
                 prefer_active: bool  = True):
        """
        Parameters
        ----------
        min_area      : minimum contour area in px**2
        ref_area      : area considered "very close"; used for dist_norm
        canny_lo/hi   : Canny thresholds (tune if detection is too noisy)
        prefer_active : if True, always return active gate over a larger inactive one
        """
        self.min_area      = min_area
        self.ref_area      = ref_area
        self.canny_lo      = canny_lo
        self.canny_hi      = canny_hi
        self.prefer_active = prefer_active
        # Dilation kernel: merges fragmented pipe-edge contours into one
        # connected gate outline, especially at close range where thick
        # polylines produce multiple separate 2-vertex contours.
        # 3x3 (not 7x7) — keeps gate outlines small enough not to bleed
        # into the sky/ground horizon band.
        self._dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect a gate in a BGR camera frame.

        Returns dict:
          gate_visible, active_visible, cx, cy, w, h, area, n_verts,
          bearing_px, elevation_px, dist_estimate (m), obs_vector (5,),
          candidates (list of _Candidate), poly (Nx2 vertices of best)
        """
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(grey, (5, 5), 1.5)
        edges = cv2.Canny(blur, self.canny_lo, self.canny_hi)
        # Erase the sky/ground horizon seam (constant full-width edge at y≈120)
        # so it cannot merge with gate contours after dilation.
        # Also erase HUD text rows at the top of the frame.
        edges[:30,  :] = 0   # top HUD text band
        edges[112:128, :] = 0  # sky/ground boundary seam
        # Dilate edges: merges the two parallel Canny edges from each thick
        # pipe wall into a single connected contour, fixing close-range dropout.
        edges = cv2.dilate(edges, self._dil_k, iterations=1)

        candidates = self._find_candidates(edges)

        if not candidates:
            # No polygon contour found — gate is too far / partially off-screen.
            # Try standalone orange-dot detection: even at 30-40 m the corner
            # LEDs contribute ~100+ orange pixels and are findable via HSV.
            # Return a coarse bearing toward the orange centroid with low
            # visibility (0.15) so the policy gets a directional nudge without
            # mistaking it for a close approach.
            orange_mask = self._orange_mask(frame)
            orange_ct   = cv2.countNonZero(orange_mask)
            if orange_ct >= 8:
                M = cv2.moments(orange_mask)
                if M['m00'] > 0:
                    ocx = int(M['m10'] / M['m00'])
                    ocy = int(M['m01'] / M['m00'])
                    obs = np.array([
                        float(np.clip((ocx - _CX) / _CX, -1.0, 1.0)),
                        float(np.clip((_CY - ocy) / _CY, -1.0, 1.0)),
                        1.0,   # dist unknown → report as far
                        0.15,  # very faint visibility — direction only, no frame info
                        0.5,   # approach angle unknown
                    ], dtype=np.float32)
                    return {
                        'gate_visible':   True,
                        'active_visible': True,
                        'cx':             ocx,
                        'cy':             ocy,
                        'w':              0,
                        'h':              0,
                        'area':           0.0,
                        'n_verts':        0,
                        'bearing_px':     ocx - _CX,
                        'elevation_px':   _CY - ocy,
                        'dist_estimate':  60.0,
                        'obs_vector':     obs,
                        'candidates':     [],
                        'poly':           None,
                    }
            return self._empty_result()

        # Tag active gate via orange corner dots
        orange_mask = self._orange_mask(frame)
        for c in candidates:
            c.is_active = self._has_orange_dots(c, orange_mask)

        best = self._pick_best(candidates)
        if best is None:
            return self._empty_result()

        bearing_px   = best.cx - _CX
        elevation_px = _CY - best.cy
        area_n       = min(best.area / self.ref_area, 1.0)
        dist_norm    = 1.0 - area_n
        dist_metres  = float(np.clip(600.0 / (np.sqrt(max(best.area, 1)) + 1e-6), 0.5, 60.0))
        # visibility: 0.0 if this is NOT the active (orange) gate — policy must
        # ignore already-passed gates and keep scanning for the orange one.
        # For the active gate, scale 0.3-1.0 by detection area so the signal
        # is stronger when the gate fills more of the frame (close / aligned).
        if best.is_active:
            visibility = float(np.clip(0.3 + 0.7 * area_n, 0.3, 1.0))
        else:
            visibility = 0.0   # wrong gate — do not reward flying toward it
        max_side     = max(best.w, best.h, 1)
        min_side     = min(best.w, best.h, 1)
        approach_ang = float(min_side / max_side)

        obs = np.array([
            float(np.clip(bearing_px   / _CX, -1.0, 1.0)),
            float(np.clip(elevation_px / _CY, -1.0, 1.0)),
            float(np.clip(dist_norm,           0.0, 1.0)),
            visibility,
            approach_ang,
        ], dtype=np.float32)

        return {
            'gate_visible':   True,
            'active_visible': best.is_active,
            'cx':             best.cx,
            'cy':             best.cy,
            'w':              best.w,
            'h':              best.h,
            'area':           best.area,
            'n_verts':        best.n_verts,
            'bearing_px':     bearing_px,
            'elevation_px':   elevation_px,
            'dist_estimate':  dist_metres,
            'obs_vector':     obs,
            'candidates':     candidates,
            'poly':           best.poly,
        }

    def draw_overlay(self, frame: np.ndarray,
                     result: Dict[str, Any]) -> np.ndarray:
        """Draw detection overlay on a copy of frame (BGR) for debugging."""
        vis = frame.copy()

        if not result['gate_visible']:
            cv2.putText(vis, 'NO GATE', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return vis

        # All candidates (faint cyan dots)
        for c in result.get('candidates', []):
            cv2.circle(vis, (c.cx, c.cy), 3, (200, 200, 0), -1)

        # Best candidate polygon (magenta outline)
        poly = result.get('poly')
        if poly is not None and len(poly) >= 3:
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], isClosed=True,
                          color=(255, 0, 255), thickness=2)

        cx, cy = result['cx'], result['cy']
        cv2.drawMarker(vis, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.line(vis, (_CX, _CY), (cx, cy), (255, 128, 0), 1)

        shape_name = 'pentagon' if result['n_verts'] == 5 else 'square'
        lines = [
            f"shape: {shape_name} ({result['n_verts']}v)",
            f"bear:  {result['bearing_px']:+d} px",
            f"elev:  {result['elevation_px']:+d} px",
            f"dist:  {result['dist_estimate']:.1f} m",
            f"act:   {'YES' if result['active_visible'] else 'no'}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (5, _IMG_H - 82 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _find_candidates(self, edges: np.ndarray) -> List[_Candidate]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        img_area = _IMG_W * _IMG_H
        candidates: List[_Candidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            if area > img_area * _MAX_AREA_RATIO:
                continue

            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, _APPROX_EPS * peri, True)
            n = len(approx)
            if n not in _VALID_VERTS:
                continue

            # Convexity check — relaxed (0.15) to allow partially off-screen
            # gates at close range whose hull-fill ratio drops.
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area > 0 and area / hull_area < 0.15:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Reject full-width/height blobs (background bleed / HUD text edges)
            if w > _IMG_W * 0.92 or h > _IMG_H * 0.92:
                continue
            # Reject extreme aspect ratios
            ar = float(w) / max(h, 1)
            if ar > 6.0 or ar < 0.17:
                continue

            cx = x + w // 2
            cy = y + h // 2
            c  = _Candidate(cx, cy, w, h, area, n,
                            approx.reshape(-1, 2).astype(np.int32))

            # Score: larger + more centred = better
            lateral_penalty = abs(cx - _CX) / _CX
            c.score = area * 0.5 + (1.0 - lateral_penalty) * 500.0
            candidates.append(c)
        return candidates

    def _orange_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary mask of orange corner LED pixels."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, _ORANGE_LO, _ORANGE_HI)

    def _has_orange_dots(self, c: _Candidate,
                         orange_mask: np.ndarray) -> bool:
        """True if candidate's bounding box contains orange dot pixels."""
        x1 = max(0, c.cx - c.w // 2)
        y1 = max(0, c.cy - c.h // 2)
        x2 = min(_IMG_W, c.cx + c.w // 2)
        y2 = min(_IMG_H, c.cy + c.h // 2)
        patch = orange_mask[y1:y2, x1:x2]
        return int(patch.sum()) > 5   # min 5 orange pixels (1 dot = ~28 px at r=3)

    def _pick_best(self, candidates: List[_Candidate]
                   ) -> Optional[_Candidate]:
        if not candidates:
            return None
        if self.prefer_active:
            active = [c for c in candidates if c.is_active]
            if active:
                return max(active, key=lambda c: c.score)
        return max(candidates, key=lambda c: c.score)

    # -----------------------------------------------------------------------
    # Empty result
    # -----------------------------------------------------------------------
    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            'gate_visible':   False,
            'active_visible': False,
            'cx':             _CX,
            'cy':             _CY,
            'w':              0,
            'h':              0,
            'area':           0.0,
            'n_verts':        0,
            'bearing_px':     0,
            'elevation_px':   0,
            'dist_estimate':  60.0,
            'obs_vector':     np.zeros(5, dtype=np.float32),
            'candidates':     [],
            'poly':           None,
        }


# ---------------------------------------------------------------------------
# Legacy compatibility shim
# ---------------------------------------------------------------------------
class GateDetector:
    """Backward-compatible wrapper; prefer SimGateDetector directly."""

    def __init__(self, config=None):
        self._det = SimGateDetector()

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        bgr = image if _looks_bgr(image) else image[:, :, ::-1]
        result = self._det.detect(bgr)
        next_gate = None
        if result['gate_visible']:
            cx, cy, w, h = result['cx'], result['cy'], result['w'], result['h']
            next_gate = {
                'center':        (cx, cy),
                'bbox':          (cx - w // 2, cy - h // 2, w, h),
                'area':          result['area'],
                'aspect_ratio':  w / max(h, 1),
                'confidence':    min(1.0, result['area'] / _REF_AREA),
                'bearing_px':    result['bearing_px'],
                'elevation_px':  result['elevation_px'],
                'dist_estimate': result['dist_estimate'],
                'n_verts':       result['n_verts'],
                'poly':          result['poly'],
                'is_active':     result['active_visible'],
            }
        return {
            'gates':        [next_gate] if next_gate else [],
            'confidence':   next_gate['confidence'] if next_gate else 0.0,
            'next_gate':    next_gate,
            'num_detected': 1 if next_gate else 0,
            'obs_vector':   result['obs_vector'],
        }

    def visualize(self, image: np.ndarray, detection_result) -> np.ndarray:
        bgr = image if _looks_bgr(image) else image[:, :, ::-1]
        return self._det.draw_overlay(bgr, detection_result)


def _looks_bgr(image: np.ndarray) -> bool:
    """Heuristic: simulator sky is blue, so B channel > R in top rows."""
    if image.shape[0] < 10 or image.ndim < 3:
        return True
    sky = image[:min(30, image.shape[0]), :, :]
    return float(sky[:, :, 0].mean()) > float(sky[:, :, 2].mean())
