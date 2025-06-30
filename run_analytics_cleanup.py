#!/usr/bin/env python3
"""
Comprehensive cleanup script for the analytics directory.
Run this script to remove all old/deprecated files and directories.
"""

import os
import shutil
import sys

def main():
    # Base directory
    base_dir = "/Users/daws/ADMF-PC/src/analytics"
    
    if not os.path.exists(base_dir):
        print(f"Error: Analytics directory not found at {base_dir}")
        sys.exit(1)
    
    # Files to remove (from cleanup_list.txt)
    files_to_remove = [
        "backtesting_framework.py",
        "calculate_log_returns.py",
        "cheap-filters.md",
        "correlation_filter.py",
        "exceptions.py",
        "execution_cost_analyzer.py",
        "fast_correlation_filter.py",
        "fixed_backtesting_example.py",
        "functions.py",
        "grid_search_analyzer.py",
        "integration.py",
        "metrics.py",
        "migration.py",
        "parameter_export.py",
        "patterns.py",
        "populate_from_metadata.py",
        "reports.py",
        "schema.py",
        "signal_context_analysis.py",
        "signal_performance_analyzer.py",
        "signal_reconstruction.py",
        "strategy_filter.py",
        "strategy_filter_optimized.py",
        "workspace.py"
    ]
    
    # Directories to remove
    directories_to_remove = [
        "cli",
        "mining",
        "queries",
        "sparse_trace_analysis",
        "storage"
    ]
    
    # Also remove __pycache__ directories
    pycache_dirs = []
    
    print("Analytics Directory Cleanup")
    print("=" * 50)
    
    # First, find all __pycache__ directories
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            pycache_dirs.append(os.path.join(root, "__pycache__"))
    
    # Remove files
    print("\nRemoving old/deprecated files:")
    print("-" * 30)
    removed_count = 0
    for file in files_to_remove:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✓ Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Error removing {file}: {e}")
        else:
            print(f"- Skipped (not found): {file}")
    
    # Remove directories
    print("\n\nRemoving old/deprecated directories:")
    print("-" * 30)
    for directory in directories_to_remove:
        dir_path = os.path.join(base_dir, directory)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Removed directory: {directory}/")
                removed_count += 1
            except Exception as e:
                print(f"✗ Error removing {directory}: {e}")
        else:
            print(f"- Skipped (not found): {directory}/")
    
    # Remove __pycache__ directories
    print("\n\nRemoving __pycache__ directories:")
    print("-" * 30)
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            rel_path = os.path.relpath(pycache_dir, base_dir)
            print(f"✓ Removed: {rel_path}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Error removing {pycache_dir}: {e}")
    
    # Remove old README.md
    old_readme = os.path.join(base_dir, "README.md")
    if os.path.exists(old_readme):
        try:
            os.remove(old_readme)
            print(f"\n✓ Removed old README.md")
            removed_count += 1
        except Exception as e:
            print(f"\n✗ Error removing README.md: {e}")
    
    # Remove cleanup_list.txt
    cleanup_list = os.path.join(base_dir, "cleanup_list.txt")
    if os.path.exists(cleanup_list):
        try:
            os.remove(cleanup_list)
            print(f"✓ Removed cleanup_list.txt")
            removed_count += 1
        except Exception as e:
            print(f"✗ Error removing cleanup_list.txt: {e}")
    
    # Rename README_NEW.md to README.md
    new_readme = os.path.join(base_dir, "README_NEW.md")
    if os.path.exists(new_readme):
        try:
            os.rename(new_readme, old_readme)
            print(f"\n✓ Renamed README_NEW.md to README.md")
        except Exception as e:
            print(f"\n✗ Error renaming README_NEW.md: {e}")
    
    # List remaining files
    print("\n\nRemaining files in analytics directory:")
    print("-" * 30)
    try:
        items = sorted(os.listdir(base_dir))
        for item in items:
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Count files in directory
                file_count = sum(1 for _ in os.listdir(item_path))
                print(f"  {item}/ ({file_count} items)")
            else:
                # Show file size
                size = os.path.getsize(item_path)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                print(f"  {item} ({size_str})")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    print(f"\n\nCleanup Summary:")
    print(f"✓ Removed {removed_count} files and directories")
    print("\nThe analytics module has been cleaned up!")
    print("\nExpected remaining files:")
    print("  - __init__.py (updated)")
    print("  - trace_analysis.py")
    print("  - pattern_discovery.py")
    print("  - trade_metrics.py")
    print("  - queries.py")
    print("  - example_usage.py")
    print("  - README.md (renamed from README_NEW.md)")
    print("  - saved_patterns/ (directory)")

if __name__ == "__main__":
    main()