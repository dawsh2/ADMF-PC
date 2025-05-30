#!/usr/bin/env python3
"""Fix callable type hints for Python 3.9+ compatibility."""

import os
import re
from pathlib import Path

def fix_file(filepath):
    """Fix callable type hints in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if file has callable[[ pattern
    if 'callable[[' not in content:
        return False
    
    # Check if Callable is already imported
    has_callable_import = 'from typing import' in content and 'Callable' in content
    
    # Replace callable[[ with Callable[[
    new_content = re.sub(r'\bcallable\[\[', 'Callable[[', content)
    
    # Add Callable to imports if needed
    if not has_callable_import and new_content != content:
        # Find the typing import line
        import_match = re.search(r'from typing import \((.*?)\)', content, re.DOTALL)
        if import_match:
            # Multi-line import
            imports = import_match.group(1)
            if 'Callable' not in imports:
                # Add Callable to the imports
                lines = imports.split('\n')
                # Find a good place to insert (alphabetically)
                for i, line in enumerate(lines):
                    if 'Callable' < line.strip().rstrip(','):
                        lines.insert(i, '    Callable,')
                        break
                else:
                    # Add at the end
                    lines[-1] = lines[-1].rstrip() + ', Callable'
                new_imports = '\n'.join(lines)
                new_content = new_content.replace(import_match.group(0), 
                                                  f'from typing import ({new_imports})')
        else:
            # Single line import
            import_match = re.search(r'from typing import (.+)', content)
            if import_match:
                imports = import_match.group(1)
                if 'Callable' not in imports:
                    new_content = new_content.replace(import_match.group(0),
                                                      f'from typing import {imports}, Callable')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    """Fix all Python files in src directory."""
    src_dir = Path('src')
    fixed_files = []
    
    for py_file in src_dir.rglob('*.py'):
        if fix_file(py_file):
            fixed_files.append(py_file)
    
    print(f"Fixed {len(fixed_files)} files:")
    for f in fixed_files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()