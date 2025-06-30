#!/usr/bin/env python3
import os
import shutil
import sys

types_dir = "/Users/daws/ADMF-PC/src/core/types"

print(f"Forcefully attempting to delete: {types_dir}")

# First, try to change permissions
try:
    for root, dirs, files in os.walk(types_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)
except:
    pass

# Now try to delete
try:
    if os.path.exists(types_dir):
        shutil.rmtree(types_dir)
        print("Success: Directory deleted")
    else:
        print("Directory doesn't exist")
except Exception as e:
    print(f"Failed with error: {e}")
    # Try alternative method
    try:
        os.system(f"rm -rf {types_dir}")
        print("Attempted with os.system")
    except Exception as e2:
        print(f"Also failed with: {e2}")

# Verify
if not os.path.exists(types_dir):
    print("✓ Confirmed: Directory has been deleted")
else:
    print("✗ Directory still exists")
    
# Clean up scripts
for script in ['delete_types_dir.py', 'delete_types.py', 'run_delete.py', 'delete_directory.py', 'force_delete_types.py']:
    try:
        os.remove(f"/Users/daws/ADMF-PC/{script}")
    except:
        pass