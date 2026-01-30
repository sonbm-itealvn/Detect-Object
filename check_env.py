"""
Script để kiểm tra .env file và environment variables
"""
import os
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] python-dotenv installed - .env file will be loaded automatically")
except ImportError:
    print("[WARNING] python-dotenv not installed")
    print("   Install with: pip install python-dotenv")
    print("   Or set environment variables directly")

print("\n" + "="*60)
print("KIEM TRA ENVIRONMENT VARIABLES")
print("="*60)

# Check YOLO_WEIGHTS_PATH
yolo_path = os.getenv("YOLO_WEIGHTS_PATH")
if yolo_path:
    print(f"\n[OK] YOLO_WEIGHTS_PATH: {yolo_path}")
    if Path(yolo_path).exists():
        print(f"   [OK] File exists")
    else:
        print(f"   [ERROR] File NOT found!")
else:
    print(f"\n[WARNING] YOLO_WEIGHTS_PATH: Not set")
    print(f"   Will use default: fine-tune.pt")

# Check FIRE_MODEL_WEIGHTS_PATH
fire_path = os.getenv("FIRE_MODEL_WEIGHTS_PATH")
if fire_path:
    print(f"\n[OK] FIRE_MODEL_WEIGHTS_PATH: {fire_path}")
    if Path(fire_path).exists():
        print(f"   [OK] File exists")
    else:
        print(f"   [ERROR] File NOT found!")
else:
    print(f"\n[WARNING] FIRE_MODEL_WEIGHTS_PATH: Not set")
    print(f"   Will try: fire-model.pt, fire_detection.pt, fire.pt")

# Check .env file
env_file = Path(".env")
if env_file.exists():
    print(f"\n[OK] .env file found: {env_file.absolute()}")
    print(f"   Reading contents...")
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"   Total lines: {len(lines)}")
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Check format issues
                        issues = []
                        if " = " in line or line.startswith(" "):
                            issues.append("Has spaces around =")
                        if "\\\\" in value and "/" not in value:
                            issues.append("Uses double backslash (use / instead)")
                        
                        status = "[OK]" if not issues else "[WARNING]"
                        print(f"   Line {i} {status}: {key}")
                        if value:
                            print(f"      Value: {value[:60]}...")
                        if issues:
                            for issue in issues:
                                print(f"      Issue: {issue}")
    except Exception as e:
        print(f"   [ERROR] Error reading .env: {e}")
else:
    print(f"\n[WARNING] .env file NOT found: {env_file.absolute()}")
    print(f"   Create .env file in project root to set environment variables")

print("\n" + "="*60)
print("TIPS")
print("="*60)
print("1. .env file format: KEY=value (no spaces around =)")
print("2. Windows paths: Use forward slashes / or single backslash \\")
print("3. Install python-dotenv: pip install python-dotenv")
print("4. Restart Python/IDE after changing .env file")

