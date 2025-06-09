#!/usr/bin/env python3
"""Move events directory to tmp and prepare for new implementation."""

import os
import shutil
from pathlib import Path

def main():
    # Get the project root
    project_root = Path("/Users/daws/ADMF-PC")
    
    # Define paths
    events_src = project_root / "src" / "core" / "events"
    events_old_dst = project_root / "tmp" / "events_old"
    
    # Create tmp/events_old directory
    print(f"Creating {events_old_dst}...")
    events_old_dst.mkdir(parents=True, exist_ok=True)
    
    # Move all contents from src/core/events to tmp/events_old
    print(f"Moving contents from {events_src} to {events_old_dst}...")
    for item in events_src.iterdir():
        dst = events_old_dst / item.name
        print(f"  Moving {item.name}...")
        shutil.move(str(item), str(dst))
    
    # Create new empty src/core/events directory (it should still exist)
    print(f"Ensuring {events_src} exists...")
    events_src.mkdir(parents=True, exist_ok=True)
    
    # Copy refactor.md and observer.md back
    files_to_copy = ["refactor.md", "observer.md"]
    for filename in files_to_copy:
        src_file = events_old_dst / filename
        dst_file = events_src / filename
        if src_file.exists():
            print(f"Copying {filename} back to events directory...")
            shutil.copy2(str(src_file), str(dst_file))
        else:
            print(f"Warning: {filename} not found in old events directory")
    
    print("\nDone! Directory structure:")
    print(f"- Old events module: {events_old_dst}")
    print(f"- New events module: {events_src}")
    print(f"- Files in new events: {list(events_src.iterdir())}")

if __name__ == "__main__":
    main()