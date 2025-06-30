#!/usr/bin/env python3
"""Fix the split print statement in the notebook"""

import json

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Fix cell 22
cell = notebook['cells'][22]
source = ''.join(cell['source'])

# Replace the problematic print statement
# The issue is that the print statement is split incorrectly
source = source.replace(
    '            print("\nWin rate by exit type:")',
    '            print("\\nWin rate by exit type:")'
)

# Update the cell source
cell['source'] = source

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("Fixed the print statement in cell 22")

# Verify the fix
with open(notebook_path, 'r') as f:
    nb = json.load(f)
    cell = nb['cells'][22]
    source = ''.join(cell['source'])
    lines = source.split('\n')
    print("\nLines 25-30 after fix:")
    for i in range(24, min(30, len(lines))):
        print(f'{i+1}: {repr(lines[i])}')