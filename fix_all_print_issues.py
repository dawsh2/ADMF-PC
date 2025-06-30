#!/usr/bin/env python3
"""Fix all print statement issues in the notebook"""

import json
import re

notebook_path = 'src/analytics/templates/trade_analysis.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

fixed_count = 0

# Go through all code cells and fix print issues
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        original_source = source
        
        # Fix common patterns of split print statements
        # Pattern 1: print("...
        # Next line: ...rest of string")
        lines = source.split('\n')
        fixed_lines = []
        j = 0
        while j < len(lines):
            line = lines[j]
            
            # Check if this line has an unterminated string in print
            if 'print(' in line and (
                (line.count('"') % 2 != 0 and not line.rstrip().endswith('"')) or
                (line.count("'") % 2 != 0 and not line.rstrip().endswith("'"))
            ):
                # Check if it's an f-string
                if 'print(f"' in line or "print(f'" in line:
                    # Look for the closing quote in next lines
                    closing_found = False
                    combined = line.rstrip()
                    for k in range(j + 1, min(j + 5, len(lines))):
                        next_line = lines[k]
                        combined += ' ' + next_line.lstrip()
                        if '"' in next_line or "'" in next_line:
                            fixed_lines.append(combined)
                            j = k
                            closing_found = True
                            fixed_count += 1
                            print(f"Fixed f-string in cell {i}, line {j+1}")
                            break
                    if not closing_found:
                        fixed_lines.append(line)
                else:
                    # Regular string
                    if j + 1 < len(lines):
                        next_line = lines[j + 1]
                        # Combine if next line looks like continuation
                        if not next_line.strip().startswith(('import', 'from', 'def', 'class', 'if', 'for', 'while')):
                            combined = line.rstrip() + ' ' + next_line.lstrip()
                            fixed_lines.append(combined)
                            j += 1
                            fixed_count += 1
                            print(f"Fixed regular string in cell {i}, line {j+1}")
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            j += 1
        
        # Update source if changed
        new_source = '\n'.join(fixed_lines)
        if new_source != original_source:
            cell['source'] = new_source

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"\nFixed {fixed_count} print statements")

# Now let's specifically check for any remaining issues
print("\nChecking for remaining issues...")
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        lines = source.split('\n')
        
        for j, line in enumerate(lines):
            # Check for print statements with odd quotes
            if 'print(' in line:
                # Count quotes
                double_quotes = line.count('"')
                single_quotes = line.count("'")
                
                # For f-strings, also check if they're properly closed
                if ('print(f"' in line or "print(f'" in line):
                    # Make sure f-string is complete on this line
                    if not (line.rstrip().endswith('")') or line.rstrip().endswith("')")):
                        if double_quotes % 2 != 0 or single_quotes % 2 != 0:
                            print(f"  Potential issue in cell {i}, line {j+1}: {line[:60]}...")