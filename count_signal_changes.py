#!/usr/bin/env python3
"""
Count signal changes from log output
"""

import re
import sys

entry_pattern = r"Entry confirmed at bar (\d+)"
exit_pattern = r"Exit at bar (\d+)"

entries = 0
exits = 0
trades = []
current_entry = None

# Read from stdin or file
for line in sys.stdin:
    entry_match = re.search(entry_pattern, line)
    exit_match = re.search(exit_pattern, line)
    
    if entry_match:
        entries += 1
        current_entry = int(entry_match.group(1))
        
    elif exit_match and current_entry:
        exits += 1
        exit_bar = int(exit_match.group(1))
        duration = exit_bar - current_entry
        trades.append(duration)
        current_entry = None

print(f"Entries: {entries}")
print(f"Exits: {exits}")
print(f"Completed trades: {len(trades)}")

if trades:
    print(f"Average duration: {sum(trades)/len(trades):.1f} bars")
    print(f"Min duration: {min(trades)} bars")
    print(f"Max duration: {max(trades)} bars")