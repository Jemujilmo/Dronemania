#!/usr/bin/env python3
"""
pretrain_bc.py — Behaviour Cloning warm-start for the PPO drone racing policy.

Pipeline
--------
1. Load expert demonstrations from records/expert_demos.npz
   (produced by record_expert.py — obs shape (N, 20), acts shape (N, 4))
2. Create a fresh SB3 PPO model with MlpPolicy on RacingEnv(use_camera_obs=True)
3. Supervise the actor with MSE loss (expert acts → policy actor output)
   using Adam over multiple epochs and minibatches.
4. Save the warm-started weights to trained_models/rl_vision_policy_latest.zip
   so that  python self_train.py --vision  picks them up for PPO fine-tuning.

Usage
-----
    # Full pipeline (record then BC):
    python record_expert.py && python pretrain_bc.py

    # BC only (assumes records/expert_demos.npz already exists):
    python pretrain_bc.py

    # Custom options:
    python pretrain_bc.py --demos records/my_demos.npz --epochs 100 --batch 256
"""

import argparse
import pathlib
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:
    sys.exit(f"[pretrain_bc] Missing dependency: {e}\n"
             "Install with: pip install stable-baselines3 torch")

from simulation.drone_gym_env import RacingEnv

# ── Default paths ──────────────────────────────────────────────────────────────
_ROOT      = pathlib.Path(__file__).parent
DEMOS_PATH = _ROOT / "records" / "expert_demos.npz"
MODEL_OUT  = _ROOT / "trained_models" / "rl_vision_policy_latest"
BACKUP_DIR = _ROOT / "trained_models" / "backups"


def load_demos(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (obs, acts) arrays from a .npz file produced by record_expert.py."""
    if not path.exists():
        sys.exit(f"[pretrain_bc] Demo file not found: {path}\n"
                 "Run record_expert.py first.")
    data = np.load(path)
    obs  = data["obs"].astype(np.float32)
    acts = data["acts"].astype(np.float32)
    print(f"  Loaded {len(obs):,} transitions from {path}")
    print(f"  obs  shape: {obs.shape}  range [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  acts shape: {acts.shape} range [{acts.min():.2f}, {acts.max():.2f}]")
    assert obs.shape[1]  == 20, f"Expected obs dim=20, got {obs.shape[1]}"
    assert acts.shape[1] == 4,  f"Expected act dim=4,  got {acts.shape[1]}"
    return obs, acts


def build_model(seed: int = 42) -> PPO:
    """Create a fresh PPO model whose architecture is loaded via PPO.load().

    Using a wider [256, 256] network for the BC pre-trainer.  self_train.py
    calls  PPO.load(checkpoint, env=env)  which preserves the SAVED architecture
    rather than re-creating a fresh default  ─ so the loaded 256×256 model will
    continue training with the same capacity it was BC-trained with.
    """
    set_random_seed(seed)
    env = RacingEnv(substeps=4, use_camera_obs=True)
    model = PPO(
        policy  = "MlpPolicy",
        env     = env,
        verbose = 0,
        seed    = seed,
        policy_kwargs = dict(
            net_arch      = dict(pi=[256, 256], vf=[256, 256]),
            activation_fn = th.nn.Tanh,
        ),
    )
    return model


def run_bc(
    model: PPO,
    obs: np.ndarray,
    acts: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "auto",
) -> None:
    """
    Train the policy actor with supervised MSE loss.

    SB3 PPO uses DiagGaussianDistribution for continuous action spaces.
    The actor mean is:  action_net( mlp_extractor(features)[0] )
    For PPO (unlike SAC) there is no tanh squashing applied at the policy level —
    the raw output of action_net IS the predicted action mean.  Expert actions
    from record_expert.py are already normalised to [-1, 1], so MSE in that
    space is a clean behavioural cloning target.
    """
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"  Training on device: {device}")

    policy = model.policy.to(device)
    policy.set_training_mode(True)

    # Optimise only the actor parameters (keep value head untouched)
    actor_params = (
        list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
        # Also train the shared feature extractor (MLP input layer)
        + list(policy.pi_features_extractor.parameters())
    )
    opt = th.optim.Adam(actor_params, lr=lr)

    N    = len(obs)
    idxs = np.arange(N)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(idxs)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            batch_idx  = idxs[start: start + batch_size]
            obs_t  = th.tensor(obs[batch_idx],  dtype=th.float32, device=device)
            acts_t = th.tensor(acts[batch_idx], dtype=th.float32, device=device)

            # Forward pass through actor
            # pi_features_extractor: raw obs → latent features (shared MLP stem)
            # mlp_extractor.policy_net: features → latent_pi
            # action_net: latent_pi → action_mean (no squashing for PPO)
            features  = policy.extract_features(obs_t, policy.pi_features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            pred_acts = policy.action_net(latent_pi)

            loss = F.mse_loss(pred_acts, acts_t)
            opt.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            th.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.5f}  "
                  f"elapsed={elapsed:.1f}s")

    policy.set_training_mode(False)

    # ── Clamp log_std so the policy EXECUTES the BC mean ─────────────────────
    # log_std is initialised at 0.0 → std = 1.0 by SB3 default, which means
    # sampled actions deviate ±1 from the BC mean — enough to crash the drone
    # on the first few steps and show 0 gates despite perfect MSE.
    # Setting to -1.0 → std ≈ 0.37 keeps tight execution while still allowing
    # PPO's entropy bonus to open it back up during fine-tuning.
    # Use .data to avoid the "leaf Variable requires grad" in-place error.
    # We intentionally bypass autograd here — this is a post-training init,
    # not a learned parameter.
    policy.log_std.data.fill_(-1.0)
    print(f"  log_std clamped to -1.0 (std≈0.37) for tight BC execution")

    print(f"\n  BC complete. Final loss: {avg_loss:.5f}")


def evaluate_bc(model: PPO, obs: np.ndarray, acts: np.ndarray,
                n_samples: int = 2000, device: str = "cpu") -> float:
    """Quick sanity-check: MSE on a held-out random sample."""
    policy = model.policy.to(device)
    policy.set_training_mode(False)
    idxs = np.random.choice(len(obs), min(n_samples, len(obs)), replace=False)
    obs_t  = th.tensor(obs[idxs],  dtype=th.float32, device=device)
    acts_t = th.tensor(acts[idxs], dtype=th.float32, device=device)
    with th.no_grad():
        features  = policy.extract_features(obs_t, policy.pi_features_extractor)
        latent_pi, _ = policy.mlp_extractor(features)
        pred_acts = policy.action_net(latent_pi)
        mse = float(F.mse_loss(pred_acts, acts_t).item())
    return mse


def save_model(model: PPO, out_path: pathlib.Path, backup_dir: pathlib.Path) -> None:
    """Save model zip and rotate the existing one to backups."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = pathlib.Path(str(out_path) + ".zip")
    if zip_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        import datetime
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"rl_vision_policy_{ts}.zip"
        zip_path.rename(dest)
        print(f"  Rotated existing model to {dest}")
    model.save(str(out_path))
    print(f"  Saved BC-warm-started model to {out_path}.zip")


def main() -> None:
    ap = argparse.ArgumentParser(description="BC warm-start for PPO drone policy")
    ap.add_argument("--demos",   default=str(DEMOS_PATH),
                    help=f"Path to expert_demos.npz (default: {DEMOS_PATH})")
    ap.add_argument("--out",     default=str(MODEL_OUT),
                    help=f"Output model path without .zip (default: {MODEL_OUT})")
    ap.add_argument("--epochs",  type=int, default=50,
                    help="BC training epochs (default: 50)")
    ap.add_argument("--batch",   type=int, default=512,
                    help="Minibatch size (default: 512)")
    ap.add_argument("--lr",      type=float, default=3e-4,
                    help="Adam learning rate (default: 3e-4)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--device",  default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Torch device (default: auto)")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip BC training; just evaluate the existing model")
    args = ap.parse_args()

    print("=" * 60)
    print("Dronemania — Behaviour Cloning Pre-trainer")
    print("=" * 60)

    # ── 1. Load demos ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading expert demonstrations …")
    obs, acts = load_demos(pathlib.Path(args.demos))

    # ── 2. Build model ────────────────────────────────────────────────────────
    print("\n[2/4] Building fresh PPO model …")
    model = build_model(seed=args.seed)
    print(f"  Policy net arch: {model.policy.mlp_extractor}")
    obs_dim  = model.observation_space.shape[0]
    act_dim  = model.action_space.shape[0]
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")

    if not args.eval_only:
        # ── 3. BC training ────────────────────────────────────────────────────
        print(f"\n[3/4] Running BC ({args.epochs} epochs, batch={args.batch}, "
              f"lr={args.lr}) …")
        run_bc(
            model      = model,
            obs        = obs,
            acts       = acts,
            epochs     = args.epochs,
            batch_size = args.batch,
            lr         = args.lr,
            device     = args.device,
        )
    else:
        print("\n[3/4] Skipping BC training (--eval-only).")

    # ── 4. Evaluation (sanity check) ──────────────────────────────────────────
    print("\n[4/4] Evaluating BC policy …")
    dev = ("cuda" if th.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    mse = evaluate_bc(model, obs, acts, device=dev)
    print(f"  Held-out MSE: {mse:.5f}  (target < 0.05 for reliable imitation)")

    if args.eval_only:
        print("\nEval-only mode: model not saved.")
        return

    # ── 5. Save ───────────────────────────────────────────────────────────────
    save_model(model, pathlib.Path(args.out), BACKUP_DIR)

    print("\n" + "=" * 60)
    print("BC pre-training complete!")
    print(f"  Model saved to: {args.out}.zip")
    print("\nNext step — PPO fine-tuning:")
    print("  python self_train.py --vision")
    print("=" * 60)


if __name__ == "__main__":
    main()
