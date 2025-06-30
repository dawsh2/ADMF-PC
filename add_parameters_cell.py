#!/usr/bin/env python3
"""Add a parameters cell to the notebook for papermill"""

import json

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Create a parameters cell
parameters_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "tags": ["parameters"]
    },
    "outputs": [],
    "source": [
        "# Parameters cell for papermill\n",
        "# These values will be overridden when the notebook is executed\n",
        "\n",
        "# Path to results directory\n",
        "results_dir = '.'\n",
        "\n",
        "# Execution cost in basis points\n",
        "execution_cost_bps = 1.0\n"
    ]
}

# Insert after the first code cell (which is the imports)
# Find the first code cell
first_code_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        first_code_idx = i
        break

if first_code_idx is not None:
    # Insert after the imports cell
    notebook['cells'].insert(first_code_idx + 1, parameters_cell)
    print(f"Added parameters cell at position {first_code_idx + 1}")
else:
    print("Could not find code cell to insert after")

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Updated notebook saved to {notebook_path}")