"""
Quick test script to verify project setup.
Run this after installing requirements to ensure everything is working.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    tests = [
        ("NumPy", "numpy"),
        ("PyTorch", "torch"),
        ("Gymnasium", "gymnasium"),
        ("Stable-Baselines3", "stable_baselines3"),
        ("OpenCV", "cv2"),
        ("YAML", "yaml"),
        ("Matplotlib", "matplotlib"),
    ]
    
    results = []
    for name, module in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            results.append(True)
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            results.append(False)
    
    return all(results)


def test_project_structure():
    """Test that project structure is correct."""
    print("\n📁 Testing project structure...")
    
    required_dirs = [
        "src/drone_interface",
        "src/simulation",
        "src/training",
        "config",
    ]
    
    results = []
    for dir_path in required_dirs:
        full_path = os.path.join(os.path.dirname(__file__), '..', dir_path)
        if os.path.exists(full_path):
            print(f"  ✅ {dir_path}")
            results.append(True)
        else:
            print(f"  ❌ {dir_path} - MISSING")
            results.append(False)
    
    return all(results)


def test_environment_creation():
    """Test that drone environment can be created."""
    print("\n🚁 Testing environment creation...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from simulation.drone_gym_env import DroneEnv
        
        config = {
            'max_episode_steps': 100,
        }
        
        env = DroneEnv(config)
        print("  ✅ Environment created successfully")
        
        obs, info = env.reset()
        print(f"  ✅ Environment reset successful")
        print(f"     Observation shape: {obs.shape}")
        print(f"     Action space: {env.action_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("🚁 DRONEMANIA PROJECT SETUP TEST")
    print("=" * 50)
    
    import_ok = test_imports()
    structure_ok = test_project_structure()
    
    print("\n" + "=" * 50)
    
    if import_ok and structure_ok:
        print("✅ All basic tests passed!")
        print("\nNext steps:")
        print("  1. Implement PyBullet simulation in src/drone_interface/sim_drone.py")
        print("  2. Define reward function in src/simulation/drone_gym_env.py")
        print("  3. Run: python src/training/train.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        if not import_ok:
            print("\n📦 Install missing packages:")
            print("   pip install -r requirements.txt")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
