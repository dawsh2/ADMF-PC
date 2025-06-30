#!/usr/bin/env python3
"""Fix all metadata parsing issues in trade_analysis.ipynb"""

import json
import re

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Fixed metadata parsing function
safe_parse_func = """def safe_parse_metadata(x):
    if pd.isna(x) or x is None:
        return {}
    elif isinstance(x, dict):
        return x
    elif isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return {}
    else:
        return {}"""

# Pattern to find problematic metadata parsing
problematic_patterns = [
    r"lambda x: json\.loads\(x\) if x else \{\}",
    r"json\.loads\(row\[col\]\) if row\[col\] else \{\}",
    r"json\.loads\(x\) if x else \{\}"
]

fixed_count = 0

# Check all code cells
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        original_source = ''.join(cell['source'])
        
        # Check if this cell has problematic metadata parsing
        for pattern in problematic_patterns:
            if re.search(pattern, original_source):
                print(f"Found problematic pattern in cell {i}")
                
                # Different fixes based on context
                if "for trace_type, df in traces.items():" in original_source:
                    # This is the main parsing loop
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
                    fixed_count += 1
                    print(f"  Fixed cell {i} (main parsing loop)")
                    
                elif "def parse_metadata" in original_source:
                    # This is the parse_metadata function definition
                    # Already has the right structure, just need to update the logic
                    # Skip this one as it's already been fixed in previous edit
                    print(f"  Cell {i} is parse_metadata function - already fixed")
                    
                else:
                    # Generic fix for other metadata parsing
                    new_source = re.sub(
                        r"lambda x: json\.loads\(x\) if x else \{\}",
                        "safe_parse_metadata",
                        original_source
                    )
                    if new_source != original_source:
                        # Add the function definition at the beginning
                        cell['source'] = [
                            "# Define safe metadata parser\n",
                            safe_parse_func + "\n\n",
                            new_source
                        ]
                        fixed_count += 1
                        print(f"  Fixed cell {i} (generic metadata parsing)")

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"\nFixed {fixed_count} cells")
print(f"Notebook saved to {notebook_path}")