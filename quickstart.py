#!/usr/bin/env python3
"""
Quick start: Test PyBullet simulator immediately
Minimal dependencies, instant feedback
"""

import subprocess
import sys
import os

def install_deps():
    """Install minimal dependencies."""
    print("📦 Installing dependencies...")
    packages = [
        "pybullet",
        "numpy", 
        "opencv-python",
        "gymnasium",
        "pyyaml",
        "simple-pid",
        "filterpy"
    ]
    
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("✅ Dependencies installed\n")

def verify_imports():
    """Verify all imports work."""
    print("🔍 Verifying imports...")
    imports = [
        ("pybullet", "PyBullet"),
        ("gymnasium", "Gymnasium"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - FAILED")
            all_ok = False
    
    if not all_ok:
        print("\n❌ Some imports failed. Run: python quickstart.py install")
        return False
    
    print("✅ All imports OK\n")
    return True

def run_test():
    """Run simulator test."""
    print("🚁 Launching simulator test...")
    print("-" * 50)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    result = subprocess.run(
        [sys.executable, "test_runner.py", "--mode", "headless", "--episodes", "1"],
        capture_output=False
    )
    
    return result.returncode == 0

def main():
    """Quick start sequence."""
    print("=" * 50)
    print("🚁 DRONEMANIA QUICKSTART")
    print("=" * 50)
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        install_deps()
    
    if not verify_imports():
        print("\n💾 Run: pip install -r requirements-new.txt")
        return 1
    
    print("Starting simulator test...\n")
    
    if run_test():
        print("\n✅ Simulator working!")
        print("\nNext steps:")
        print("  1. Read SIMULATOR_SETUP.md for more info")
        print("  2. Check config/autonomy.yaml for tuning")
        print("  3. Integrate full autonomy stack")
        return 0
    else:
        print("\n❌ Simulator test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
