import os
import glob

# Find and remove backup files
backup_files = glob.glob('/Users/daws/ADMF-PC/src/core/coordinator_refactor/*.bak')
for file in backup_files:
    try:
        os.remove(file)
        print(f"Removed: {file}")
    except Exception as e:
        print(f"Failed to remove {file}: {e}")

print(f"\nTotal backup files removed: {len(backup_files)}")
