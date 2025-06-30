#!/usr/bin/env python3
"""Clean up old/deprecated files from analytics directory."""

import os
import shutil

# Base directory
base_dir = "/Users/daws/ADMF-PC/src/analytics"

# Files to remove
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

# Remove files
print("Removing old/deprecated files...")
for file in files_to_remove:
    file_path = os.path.join(base_dir, file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed: {file}")
    else:
        print(f"File not found: {file}")

# Remove directories
print("\nRemoving old/deprecated directories...")
for directory in directories_to_remove:
    dir_path = os.path.join(base_dir, directory)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed directory: {directory}")
    else:
        print(f"Directory not found: {directory}")

# Also remove the old README.md and cleanup_list.txt
old_readme = os.path.join(base_dir, "README.md")
if os.path.exists(old_readme):
    os.remove(old_readme)
    print(f"\nRemoved old README.md")

cleanup_list = os.path.join(base_dir, "cleanup_list.txt")
if os.path.exists(cleanup_list):
    os.remove(cleanup_list)
    print(f"Removed cleanup_list.txt")

# Rename README_NEW.md to README.md
new_readme = os.path.join(base_dir, "README_NEW.md")
if os.path.exists(new_readme):
    os.rename(new_readme, old_readme)
    print(f"\nRenamed README_NEW.md to README.md")

print("\nCleanup complete!")

# List remaining files
print("\nRemaining files in analytics directory:")
for item in sorted(os.listdir(base_dir)):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        print(f"  {item}/")
    else:
        print(f"  {item}")