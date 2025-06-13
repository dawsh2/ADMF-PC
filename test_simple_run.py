#!/usr/bin/env python3
"""Test a simple run to verify workspace storage"""

import logging
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create simple test config
test_config = {
    "data": {
        "source": "yahoo",
        "symbols": ["SPY"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "data_dir": "./data"
    },
    "strategies": [
        {
            "name": "simple_momentum",
            "type": "simple_momentum", 
            "params": {
                "lookback_period": 20,
                "momentum_threshold": 0.01
            }
        }
    ],
    "execution": {
        "commission": 0.001,
        "slippage": 0.0005,
        "initial_capital": 100000,
        "position_size": 0.95
    }
}

# Save config
config_path = Path("test_simple_config.yaml")
with open(config_path, 'w') as f:
    yaml.dump(test_config, f)

logger.info(f"Created test config at {config_path}")

# Run using venv Python
import subprocess
cmd = [
    "./venv/bin/python", "main.py",
    "--config", str(config_path),
    "--signal-generation",
    "--bars", "100"  # Limit to 100 bars for quick test
]

logger.info(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

# Check output
if result.returncode == 0:
    logger.info("✓ Run completed successfully")
    # Check for workspace creation
    if "SQL analytics workspace created" in result.stdout:
        logger.info("✓ Analytics workspace was created!")
        # Extract workspace name
        import re
        match = re.search(r"SQL analytics workspace created: ([^\s]+)", result.stdout)
        if match:
            workspace_name = match.group(1)
            logger.info(f"✓ Workspace: {workspace_name}")
    else:
        logger.warning("✗ No workspace creation message found")
        logger.info("STDOUT:")
        print(result.stdout)
        if result.stderr:
            logger.info("STDERR:")
            print(result.stderr)
else:
    logger.error(f"✗ Run failed with return code {result.returncode}")
    logger.error("STDERR:")
    print(result.stderr)

# Cleanup
config_path.unlink()