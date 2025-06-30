#!/usr/bin/env python3
"""Fix cell types in the notebook"""

import json

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Fix cells that should be markdown but are marked as code
fixes_made = 0
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        
        # Check if this looks like markdown content
        if source.strip().startswith('###') or (
            'Validate that stop losses' in source and 
            'def' not in source and 
            'import' not in source
        ):
            print(f"Fixing cell {i}: converting from code to markdown")
            print(f"  Content: {source[:100]}...")
            cell['cell_type'] = 'markdown'
            # Remove code-specific fields
            if 'execution_count' in cell:
                del cell['execution_count']
            if 'outputs' in cell:
                del cell['outputs']
            fixes_made += 1

print(f"\nFixed {fixes_made} cells")

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Saved fixed notebook to {notebook_path}")