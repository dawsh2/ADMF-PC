#!/usr/bin/env python3
import shutil
import os

types_dir = "/Users/daws/ADMF-PC/src/core/types"

if os.path.exists(types_dir):
    shutil.rmtree(types_dir)
    print(f"Successfully deleted {types_dir}")
else:
    print(f"Directory {types_dir} does not exist")