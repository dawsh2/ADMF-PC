#!/usr/bin/env python3
"""Script to delete the old types directory."""

import os
import shutil

types_dir = "/Users/daws/ADMF-PC/src/core/types"

if os.path.exists(types_dir):
    print(f"Deleting {types_dir}...")
    shutil.rmtree(types_dir)
    print("Directory deleted successfully!")
else:
    print(f"Directory {types_dir} does not exist.")

# Verify deletion
if not os.path.exists(types_dir):
    print("✓ Confirmed: types directory has been removed")
else:
    print("✗ Error: types directory still exists")