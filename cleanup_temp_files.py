#!/usr/bin/env python3
"""Remove temporary cleanup scripts."""

import os

files_to_remove = [
    "/Users/daws/ADMF-PC/cleanup_analytics.py",
    "/Users/daws/ADMF-PC/test_shell.py",
    "/Users/daws/ADMF-PC/cleanup_temp_files.py"  # Remove itself
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")

print("Temporary files cleaned up!")