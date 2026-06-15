#!/usr/bin/env python3
"""
Gate detector visual test - pentagon/square shape-based detection.

Controls:  A = toggle auto-advance   SPACE = manual step   Q/ESC = quit
Flags:     --headless    no window, batch stats
           --cam-obs     print obs_vector each step
           --steps N     steps in headless mode (default 500)
"""
import argparse, sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import cv2
from simulation.racing_simulator import RacingSimulator
from perception.gate_detector   import SimGateDetector

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--cam-obs",  action="store_true")
ap.add_argument("--steps",    type=int, default=500)
args = ap.parse_args()

sim = RacingSimulator({"sim_rate": 60})
sim.reset()
det = SimGateDetector()

print(f"FOV: {sim.fov_degrees} deg | Gates: {len(sim.gates)}")
print(f"Gate shapes (first 6): {[g.shape for g in sim.gates[:6]]}")
print()

def fly_toward_gate():
    state = sim.physics.get_state()
    gi    = sim.current_gate_idx
    if gi < len(sim.gates):
        to_gate = sim.gates[gi].position - state["position"]
        d       = np.linalg.norm(to_gate) + 1e-6
        fwd     = to_gate / d
    else:
        fwd = np.array([1., 0., 0.])
    sim.physics.set_state(
        position         = state["position"] + fwd * 0.20,
        velocity         = fwd * 4.0,
        orientation      = state["orientation"],
        angular_velocity = np.zeros(3))
    sim.check_gates()
    if sim.current_gate_idx >= len(sim.gates):
        sim.reset()

if args.headless:
    detected = active = 0
    for step in range(args.steps):
        fly_toward_gate()
        frame = sim.get_observation()["camera"]   # BGR
        r     = det.detect(frame)
        if r["gate_visible"]:   detected += 1
        if r["active_visible"]: active   += 1
        if step % 100 == 0:
            print(f"  step {step:4d}  vis={r['gate_visible']}  act={r['active_visible']}"
                  f"  verts={r['n_verts']}  bear={r['bearing_px']:+d}px"
                  f"  dist={r['dist_estimate']:.1f}m")
            if args.cam_obs: print(f"  obs: {r['obs_vector'].round(3)}")
    print(f"\nDetection: {detected/args.steps*100:.1f}%   Active: {active/args.steps*100:.1f}%")
    sys.exit(0)

WIN = "Gate Detector (A=auto  SPACE=step  Q=quit)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 640, 480)
auto = True
step = 0
t_last = time.time()

while True:
    if auto or step == 0:
        fly_toward_gate()
        step += 1

    frame  = sim.get_observation()["camera"]   # BGR
    result = det.detect(frame)
    overlay = det.draw_overlay(frame, result)
    disp    = cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_NEAREST)

    gi = sim.current_gate_idx
    lines = [
        f"Step:{step}  Gate:{gi}/{len(sim.gates)}  Auto:{'ON' if auto else 'OFF'}",
        f"Shape:{sim.gates[gi].shape if gi<len(sim.gates) else '-'}  "
        f"Verts:{result['n_verts']}  Active:{'YES' if result['active_visible'] else 'no'}",
    ]
    if result['gate_visible']:
        lines.append(f"Bear:{result['bearing_px']:+d}px  "
                     f"Elev:{result['elevation_px']:+d}px  "
                     f"Dist:{result['dist_estimate']:.1f}m")
        if args.cam_obs:
            v = result['obs_vector']
            lines.append(f"obs:[{v[0]:+.2f} {v[1]:+.2f} "
                         f"{v[2]:.2f} {v[3]:.2f} {v[4]:.2f}]")
    else:
        lines.append("NOT DETECTED")

    for i, ln in enumerate(lines):
        y = 20 + i*22
        cv2.putText(disp, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(disp, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow(WIN, disp)
    wait = max(1, int((0.05 - (time.time()-t_last))*1000)) if auto else 0
    key  = cv2.waitKey(wait) & 0xFF
    t_last = time.time()

    if key in (ord("q"), 27):   break
    elif key == ord("a"):       auto = not auto; print(f"Auto: {'ON' if auto else 'OFF'}")
    elif key == ord(" "): pass

cv2.destroyAllWindows()
print("Done.")
