#!/usr/bin/env python3
"""
Apply the migration by replacing old files with new clean implementations.

Run test_new_coordinator.py first to verify everything works!
"""

import shutil
from pathlib import Path
from datetime import datetime


def apply_migration():
    """Apply the migration."""
    print("=== Applying Clean Architecture Migration ===\n")
    
    coordinator_dir = Path("src/core/coordinator")
    
    # Files to replace
    replacements = [
        ("topology_clean.py", "topology.py"),
        ("coordinator_clean.py", "coordinator.py")
    ]
    
    for new_file, old_file in replacements:
        new_path = coordinator_dir / new_file
        old_path = coordinator_dir / old_file
        
        if new_path.exists():
            print(f"Replacing {old_file} with {new_file}")
            if old_path.exists():
                # Create backup just in case
                backup_path = coordinator_dir / f"{old_file}.bak"
                shutil.copy2(old_path, backup_path)
            
            # Replace file
            shutil.copy2(new_path, old_path)
            
            # Remove clean version
            new_path.unlink()
            print(f"✓ Replaced {old_file}")
        else:
            print(f"⚠️  {new_file} not found")
    
    print("\n=== Migration Applied ===")
    print("Old files backed up with .bak extension")
    print("Test the system to ensure everything works!")


if __name__ == "__main__":
    response = input("Have you run test_new_coordinator.py and verified it works? (y/n): ")
    if response.lower() == 'y':
        apply_migration()
    else:
        print("Please run test_new_coordinator.py first!")
