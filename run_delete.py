#!/usr/bin/env python3
import os
import shutil

# Get the absolute path
types_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "core", "types")

print(f"Attempting to delete: {types_dir}")

if os.path.exists(types_dir):
    try:
        shutil.rmtree(types_dir)
        print(f"Successfully deleted {types_dir}")
        
        # Verify deletion
        if not os.path.exists(types_dir):
            print("Verified: Directory has been deleted.")
        else:
            print("WARNING: Directory still exists after deletion attempt.")
    except Exception as e:
        print(f"Error deleting directory: {e}")
else:
    print(f"Directory {types_dir} does not exist")

# Clean up the helper scripts
for script in ['delete_types_dir.py', 'delete_types.py', 'run_delete.py']:
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if os.path.exists(script_path):
        try:
            os.remove(script_path)
            print(f"Cleaned up: {script}")
        except:
            pass