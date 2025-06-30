#!/usr/bin/env python3
"""Trace the decision flow to understand why entry_signal isn't stored."""

import subprocess
import sys

# Add logging to portfolio state
print("=== Adding debug logging to trace decision flow ===\n")

portfolio_file = "src/portfolio/state.py"

# Read the file
with open(portfolio_file, 'r') as f:
    content = f.read()

# Add debug logging before the entry_signal code
if "DECISION_DEBUG" not in content:
    debug_code = '''
        # DECISION_DEBUG - Temporary logging
        logger.info(f"[DECISION_DEBUG] Processing decision: type={decision.get('type')}, action={decision.get('action')}")
        logger.info(f"[DECISION_DEBUG] Full decision: {decision}")
        '''
    
    # Find where to insert
    insert_pos = content.find("# For entry orders, store the current signal value")
    if insert_pos > 0:
        # Go back to find the start of the line
        line_start = content.rfind('\n', 0, insert_pos) + 1
        indent = len(content[line_start:insert_pos])
        
        new_content = content[:insert_pos] + debug_code + '\n' + ' ' * indent + content[insert_pos:]
        
        # Write back
        with open(portfolio_file, 'w') as f:
            f.write(new_content)
        
        print("✓ Added debug logging to portfolio state")
    else:
        print("❌ Could not find insertion point")
else:
    print("Debug logging already present")

print("\nNow run: PYTHONDONTWRITEBYTECODE=1 python3 main.py --config config/bollinger/test.yaml --universal --dataset test")
print("Then check the logs for [DECISION_DEBUG] to see what decisions are being processed")