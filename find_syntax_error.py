#!/usr/bin/env python3
"""Find and fix syntax errors in notebook"""

import json
import re

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# The error is at cell 11 during execution, line 27
# Since we added a parameters cell at position 2, the original cell numbers shifted
# So execution cell 11 is probably around original cell 12-13

print("Looking for problematic cells...")

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        lines = source.split('\n')
        
        # Look for print statements that might have issues
        for line_num, line in enumerate(lines):
            # Check for print with literal newline that might not be properly quoted
            if 'print("' in line and ('\\n' in line or '\n' in line):
                # Check if this might be around line 27
                if 20 <= line_num <= 35:
                    print(f"\nFound potential issue in cell {i}, line {line_num + 1}:")
                    print(f"  {repr(line)}")
                    
                    # Check for specific pattern that might cause issues
                    if 'print("\\n' in line and not (line.rstrip().endswith('"') or line.rstrip().endswith("'")):
                        print("  -> This line might have an unterminated string!")

# Also check for cells with exactly the error pattern
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Win rate by exit type:' in source:
            print(f"\nFound cell {i} with 'Win rate by exit type'")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'print' in line:
                    print(f"  Line {j+1}: {repr(line[:80])}...")