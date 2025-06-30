#!/usr/bin/env python3
"""Delete the old utils.py file."""

import os

utils_file = "/Users/daws/ADMF-PC/src/execution/utils.py"

if os.path.exists(utils_file):
    os.remove(utils_file)
    print(f"✓ Deleted {utils_file}")
else:
    print(f"File {utils_file} does not exist")

# Verify
if not os.path.exists(utils_file):
    print("✓ Confirmed: utils.py has been removed")
    print("✓ calc.py is now the home for financial calculations")