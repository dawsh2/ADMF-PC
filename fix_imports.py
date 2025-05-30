#!/usr/bin/env python3
"""Fix import issues in the codebase."""

import os
import re

fixes = [
    # Fix Capability import in risk/capabilities.py
    {
        'file': 'src/risk/capabilities.py',
        'old': 'from ..core.infrastructure.capabilities import Capability',
        'new': 'from ..core.components.protocols import Capability'
    },
    # Fix Capability import in execution/capabilities.py  
    {
        'file': 'src/execution/capabilities.py',
        'old': 'from ..core.infrastructure.capabilities import Capability',
        'new': 'from ..core.components.protocols import Capability'
    },
    # Export SignalType and OrderSide from risk module
    {
        'file': 'src/risk/__init__.py',
        'old': '    # Types\n    "Signal",',
        'new': '    # Types\n    "Signal",\n    "SignalType",\n    "OrderSide",'
    },
    # Add missing imports to risk __init__.py
    {
        'file': 'src/risk/__init__.py',
        'old': 'from .protocols import (\n    RiskPortfolioProtocol,',
        'new': 'from .protocols import (\n    RiskPortfolioProtocol,\n    SignalType,\n    OrderSide,'
    }
]

print("Fixing import issues...")

for fix in fixes:
    filepath = fix['file']
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        if fix['old'] in content:
            new_content = content.replace(fix['old'], fix['new'])
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"✓ Fixed {filepath}")
        else:
            print(f"- Skipped {filepath} (already fixed or pattern not found)")
    else:
        print(f"✗ File not found: {filepath}")

print("\nDone! You can now run the tests.")