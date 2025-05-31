#!/usr/bin/env python3
"""Minimal backtest test to isolate hanging issue."""

import sys
import logging
sys.path.append('/Users/daws/ADMF-PC')

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    print("🧪 Testing minimal backtest components...")
    
    # Test imports
    print("1. Testing imports...")
    from src.core.coordinator.coordinator import TradingCoordinator
    from src.core.config.schemas import WorkflowConfig
    print("   ✅ Core imports successful")
    
    # Test config loading
    print("2. Testing config loading...")
    config_path = "config/spy_momentum_backtest.yaml"
    import yaml
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    print(f"   ✅ Config loaded: {raw_config['name']}")
    
    print("3. Testing coordinator creation...")
    coordinator = TradingCoordinator(enable_composable=True)
    print("   ✅ Coordinator created")
    
    print("✅ Basic components work - issue likely in execution flow")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()