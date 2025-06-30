#!/usr/bin/env python3
"""Clean up notebook structure to remove invalid outputs from markdown cells"""

import json

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Clean all cells
for cell in notebook['cells']:
    # Remove outputs from markdown cells
    if cell['cell_type'] == 'markdown' and 'outputs' in cell:
        del cell['outputs']
    
    # Ensure code cells have outputs field (even if empty)
    if cell['cell_type'] == 'code' and 'outputs' not in cell:
        cell['outputs'] = []
    
    # Ensure execution_count exists for code cells
    if cell['cell_type'] == 'code' and 'execution_count' not in cell:
        cell['execution_count'] = None

# Save the cleaned notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Cleaned notebook saved to {notebook_path}")

# Verify the structure
with open(notebook_path, 'r') as f:
    nb = json.load(f)
    
print(f"\nNotebook structure:")
print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    print(f"  Cell {i}: {cell['cell_type']}", end="")
    if 'outputs' in cell:
        print(f" (has outputs field)")
    else:
        print(f" (no outputs field)")