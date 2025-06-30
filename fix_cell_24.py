#!/usr/bin/env python3
"""Fix the specific f-string issues in cell 24"""

import json

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get cell 24
cell = notebook['cells'][24]
source = ''.join(cell['source'])

# Fix the split f-strings
# First one: print(f"\nData Quality:")
source = source.replace('print(f"\nData Quality:")', 'print(f"\\nData Quality:")')

# Second one: print(f"\nKey Findings:")
source = source.replace('print(f"\nKey Findings:")', 'print(f"\\nKey Findings:")')

# Also check if there are any other split print statements
lines = source.split('\n')
print("Checking lines in cell 24:")
for i, line in enumerate(lines):
    if 'print(f"' in line and not (line.rstrip().endswith('")') or line.rstrip().endswith('")')):
        print(f"  Line {i+1}: {repr(line)}")

# Update the cell
cell['source'] = source

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("\nFixed cell 24")

# Verify the fix
with open(notebook_path, 'r') as f:
    nb = json.load(f)
    cell = nb['cells'][24]
    source = ''.join(cell['source'])
    lines = source.split('\n')
    print("\nFirst 10 lines after fix:")
    for i in range(min(10, len(lines))):
        print(f'{i+1}: {repr(lines[i])}')