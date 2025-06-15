#!/usr/bin/env python3
"""Fix the topology.py file by removing orphaned parsing code."""

import re

# Read the file
with open('/Users/daws/ADMF-PC/src/core/coordinator/topology.py', 'r') as f:
    lines = f.readlines()

# Find the line with our new method call
new_method_line = -1
for i, line in enumerate(lines):
    if 'self._create_feature_config_from_id(feature_id)' in line:
        new_method_line = i
        break

# Find the line with "Add feature configs to context config"
add_configs_line = -1
for i, line in enumerate(lines):
    if '# Add feature configs to context config' in line:
        add_configs_line = i
        break

print(f"Found new method call at line {new_method_line + 1}")
print(f"Found add configs comment at line {add_configs_line + 1}")

if new_method_line >= 0 and add_configs_line >= 0:
    # Keep everything up to and including the new method line
    # Then skip to the add configs line
    new_lines = lines[:new_method_line + 1]
    
    # Add a blank line
    new_lines.append('\n')
    
    # Add everything from add configs line onward
    new_lines.extend(lines[add_configs_line:])
    
    # Write back
    with open('/Users/daws/ADMF-PC/src/core/coordinator/topology.py', 'w') as f:
        f.writelines(new_lines)
    
    print(f"Removed {add_configs_line - new_method_line - 1} lines of orphaned code")
    print("File fixed!")
else:
    print("Could not find the lines to fix")