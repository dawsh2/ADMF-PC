#!/usr/bin/env python3
"""Fix the metadata parsing in trade_analysis.ipynb"""

import json

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find and fix the problematic cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and any('Parse JSON metadata for all trace types' in line for line in cell['source']):
        print(f"Found problematic cell at index {i}")
        
        # Replace with fixed code
        cell['source'] = [
            "# Parse JSON metadata for all trace types\n",
            "for trace_type, df in traces.items():\n",
            "    if 'metadata' in df.columns and len(df) > 0:\n",
            "        try:\n",
            "            # Parse metadata - handle both dict and string types\n",
            "            def safe_parse_metadata(x):\n",
            "                if pd.isna(x) or x is None:\n",
            "                    return {}\n",
            "                elif isinstance(x, dict):\n",
            "                    return x\n",
            "                elif isinstance(x, str):\n",
            "                    try:\n",
            "                        return json.loads(x)\n",
            "                    except:\n",
            "                        return {}\n",
            "                else:\n",
            "                    return {}\n",
            "            \n",
            "            metadata_parsed = df['metadata'].apply(safe_parse_metadata)\n",
            "            metadata_df = pd.DataFrame(list(metadata_parsed))\n",
            "            \n",
            "            # Add parsed columns to original dataframe\n",
            "            for col in metadata_df.columns:\n",
            "                if col not in df.columns:\n",
            "                    df[col] = metadata_df[col]\n",
            "            \n",
            "            traces[trace_type] = df  # Update with parsed data\n",
            "            print(f\"Parsed {trace_type} metadata: {list(metadata_df.columns)[:10]}...\")  # Show first 10 cols\n",
            "        except Exception as e:\n",
            "            print(f\"Error parsing {trace_type} metadata: {e}\")"
        ]
        print("Fixed the cell!")
        break

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook saved to {notebook_path}")