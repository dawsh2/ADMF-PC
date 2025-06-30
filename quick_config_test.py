#!/usr/bin/env python3
"""Quick test of all configs to verify they run."""

import subprocess
import sys
from pathlib import Path
from collections import defaultdict

def test_config(config_path):
    """Test if config runs without error."""
    cmd = [
        sys.executable,
        "main.py",
        "--config", str(config_path),
        "--signal-generation",
        "--bars", "50"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Check exit code
        if result.returncode != 0:
            return False, "Non-zero exit code"
        
        # Check for Python errors
        if "Traceback" in result.stderr:
            return False, "Python exception"
        
        # Check for signals
        has_signals = "📡 SIGNAL" in result.stdout
        
        return True, "Has signals" if has_signals else "No signals"
        
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    """Test all configs."""
    config_root = Path("config/indicators")
    configs = list(config_root.rglob("*.yaml"))
    configs.sort()
    
    print(f"Testing {len(configs)} configs...")
    print("=" * 60)
    
    results = defaultdict(lambda: {"success": 0, "failed": 0, "signals": 0})
    
    for config in configs:
        category = config.parent.name
        success, msg = test_config(config)
        
        if success:
            results[category]["success"] += 1
            if "signals" in msg:
                results[category]["signals"] += 1
            status = "✅"
        else:
            results[category]["failed"] += 1
            status = "❌"
        
        print(f"{status} {config.stem:<40} {msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY BY CATEGORY:")
    print("=" * 60)
    
    total_success = 0
    total_failed = 0
    total_signals = 0
    
    for category in sorted(results.keys()):
        r = results[category]
        total = r["success"] + r["failed"]
        print(f"\n{category.upper()}:")
        print(f"  Total: {total}")
        print(f"  Success: {r['success']} ({r['success']/total*100:.0f}%)")
        print(f"  With signals: {r['signals']}")
        print(f"  Failed: {r['failed']}")
        
        total_success += r["success"]
        total_failed += r["failed"] 
        total_signals += r["signals"]
    
    print(f"\nOVERALL:")
    print(f"  Total configs: {len(configs)}")
    print(f"  Successful: {total_success} ({total_success/len(configs)*100:.0f}%)")
    print(f"  With signals: {total_signals} ({total_signals/len(configs)*100:.0f}%)")
    print(f"  Failed: {total_failed}")

if __name__ == "__main__":
    main()