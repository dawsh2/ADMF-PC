#!/usr/bin/env python3
"""Fix notebook template issues."""

import json

# Read the notebook
with open('src/analytics/templates/trade_analysis_simple.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown':
        # Remove outputs field from markdown cells
        if 'outputs' in cell:
            del cell['outputs']
    elif cell['cell_type'] == 'code':
        # Add execution_count to code cells
        if 'execution_count' not in cell:
            cell['execution_count'] = None
        # Ensure outputs is a list
        if 'outputs' not in cell:
            cell['outputs'] = []

# Save fixed notebook
with open('src/analytics/templates/trade_analysis_simple.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Fixed notebook template")