#!/usr/bin/env python3
"""Create a properly formatted cell for metadata parsing"""

import json

# Read current notebook
with open('src/analytics/templates/trade_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell after "## 2. Parse Metadata" markdown
parse_metadata_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '## 2. Parse Metadata' in ''.join(cell.get('source', [])):
        parse_metadata_index = i + 1
        break

if parse_metadata_index is None:
    # Look for alternative patterns
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and 'Parse Metadata' in ''.join(cell.get('source', [])):
            parse_metadata_index = i + 1
            print(f"Found Parse Metadata section at index {i}")
            break

if parse_metadata_index is not None:
    # Create the new cell with proper metadata parsing
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
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
    }
    
    # Check what's currently at that index
    current_cell = notebook['cells'][parse_metadata_index]
    print(f"Current cell at index {parse_metadata_index}:")
    print(f"  Type: {current_cell['cell_type']}")
    if current_cell['cell_type'] == 'code':
        preview = ''.join(current_cell.get('source', []))[:200]
        print(f"  Content preview: {preview}")
        
        # Replace it if it has the old parsing code
        if 'json.loads(x) if x else' in ''.join(current_cell.get('source', [])):
            notebook['cells'][parse_metadata_index] = new_cell
            print("  -> Replaced with fixed version")
        else:
            print("  -> Cell doesn't have the old pattern")
    else:
        # Insert the new cell
        notebook['cells'].insert(parse_metadata_index, new_cell)
        print("  -> Inserted new cell")
    
    # Save
    with open('src/analytics/templates/trade_analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("Notebook updated successfully")
else:
    print("Could not find Parse Metadata section!")