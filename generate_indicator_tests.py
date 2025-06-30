#!/usr/bin/env python3
"""
Generate test files for all indicator strategies.

This script creates comprehensive test files for strategies in src/strategy/strategies/indicators/
following the established testing patterns.
"""

import os
import sys
import ast
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_strategy_functions(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a Python file to extract strategy function signatures."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    strategies = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if this function has @strategy decorator
            has_strategy_decorator = False
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and 
                    isinstance(decorator.func, ast.Name) and 
                    decorator.func.id == 'strategy'):
                    has_strategy_decorator = True
                    break
            
            if has_strategy_decorator:
                # Extract function info
                strategy_info = {
                    'name': node.name,
                    'args': [],
                    'defaults': []
                }
                
                # Extract arguments
                for arg in node.args.args:
                    if arg.arg != 'self':  # Skip self for methods
                        strategy_info['args'].append(arg.arg)
                
                # Extract default values (simplified)
                if node.args.defaults:
                    strategy_info['defaults'] = ['...'] * len(node.args.defaults)
                
                strategies.append(strategy_info)
    
    return strategies

def generate_test_content(module_name: str, strategies: List[Dict[str, Any]]) -> str:
    """Generate test file content for a module's strategies."""
    
    test_content = f'''"""
Unit tests for {module_name} indicator strategies.

Tests all strategies in src/strategy/strategies/indicators/{module_name}.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.strategy.strategies.indicators.{module_name} import *
from src.strategy.types import Signal

class Test{module_name.title().replace("_", "")}Strategies(unittest.TestCase):
    """Test all {module_name} indicator strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with various market conditions
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Base price with trend
        base_price = 100
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 2, 100)
        
        self.prices = base_price + trend + noise
        
        # Create DataFrame with OHLCV data
        self.data = pd.DataFrame({{
            'timestamp': dates,
            'open': self.prices * 0.99,
            'high': self.prices * 1.01,
            'low': self.prices * 0.98,
            'close': self.prices,
            'volume': np.random.randint(1000000, 2000000, 100)
        }})
        
        # Add some indicators that strategies might need
        self.data['returns'] = self.data['close'].pct_change()
        
    def _test_strategy_basic(self, strategy_func, **kwargs):
        """Basic test template for any strategy function."""
        # Test with valid data
        signal = strategy_func(self.data, **kwargs)
        
        # Check signal is valid
        self.assertIsInstance(signal, Signal)
        self.assertIn(signal.direction, [-1, 0, 1])
        self.assertIsInstance(signal.magnitude, (int, float))
        self.assertGreaterEqual(signal.magnitude, 0)
        self.assertLessEqual(signal.magnitude, 1)
        
        # Test with empty data
        empty_df = pd.DataFrame()
        signal = strategy_func(empty_df, **kwargs)
        self.assertEqual(signal.direction, 0)
        
        # Test with insufficient data
        small_df = self.data.head(2)
        signal = strategy_func(small_df, **kwargs)
        self.assertEqual(signal.direction, 0)
        
        return signal
    
    def _test_strategy_edge_cases(self, strategy_func, **kwargs):
        """Test edge cases for a strategy."""
        # Test with NaN values
        data_with_nan = self.data.copy()
        data_with_nan.loc[50:55, 'close'] = np.nan
        signal = strategy_func(data_with_nan, **kwargs)
        self.assertIsInstance(signal, Signal)
        
        # Test with extreme values
        extreme_data = self.data.copy()
        extreme_data['close'] = extreme_data['close'] * 1000
        signal = strategy_func(extreme_data, **kwargs)
        self.assertIsInstance(signal, Signal)
        
        # Test with zero volume
        zero_vol_data = self.data.copy()
        zero_vol_data['volume'] = 0
        signal = strategy_func(zero_vol_data, **kwargs)
        self.assertIsInstance(signal, Signal)
'''

    # Add specific tests for each strategy
    for strategy in strategies:
        # Create default parameters based on common patterns
        default_params = _get_default_params_for_strategy(strategy['name'])
        
        test_content += f'''
    
    def test_{strategy['name']}(self):
        """Test {strategy['name']} strategy."""
        # Test with default parameters
        signal = self._test_strategy_basic({strategy['name']}, **{default_params})
        
        # Test specific conditions for this strategy
        self._test_{strategy['name']}_conditions()
        
        # Test edge cases
        self._test_strategy_edge_cases({strategy['name']}, **{default_params})
    
    def _test_{strategy['name']}_conditions(self):
        """Test specific market conditions for {strategy['name']}."""
        # Create specific market conditions based on strategy type
        {_generate_specific_tests(strategy['name'])}
'''

    test_content += '''

if __name__ == '__main__':
    unittest.main()
'''
    
    return test_content

def _get_default_params_for_strategy(strategy_name: str) -> dict:
    """Get reasonable default parameters for a strategy based on its name."""
    # Common parameter patterns
    if 'ma' in strategy_name or 'moving_average' in strategy_name:
        return {'fast_period': 10, 'slow_period': 20}
    elif 'rsi' in strategy_name:
        return {'period': 14, 'oversold': 30, 'overbought': 70}
    elif 'momentum' in strategy_name:
        return {'lookback_period': 20, 'threshold': 0.02}
    elif 'bollinger' in strategy_name or 'bb' in strategy_name:
        return {'period': 20, 'num_std': 2}
    elif 'volume' in strategy_name:
        return {'lookback_period': 20, 'threshold': 1.5}
    elif 'breakout' in strategy_name:
        return {'lookback_period': 20, 'breakout_factor': 1.01}
    elif 'support' in strategy_name or 'resistance' in strategy_name:
        return {'lookback_period': 50, 'touch_threshold': 0.02}
    else:
        return {}

def _generate_specific_tests(strategy_name: str) -> str:
    """Generate specific test conditions based on strategy type."""
    if 'crossover' in strategy_name:
        return '''# Test bullish crossover
        bullish_data = self.data.copy()
        bullish_data['close'] = np.concatenate([
            np.linspace(100, 90, 50),  # Downtrend
            np.linspace(90, 110, 50)   # Uptrend
        ])
        signal = {0}(bullish_data)
        # Should generate buy signal during crossover'''.format(strategy_name)
    
    elif 'momentum' in strategy_name:
        return '''# Test strong momentum
        momentum_data = self.data.copy()
        momentum_data['close'] = 100 * (1.1 ** np.arange(100))  # Strong uptrend
        signal = {0}(momentum_data)
        self.assertEqual(signal.direction, 1)  # Should be bullish'''.format(strategy_name)
    
    elif 'mean_reversion' in strategy_name:
        return '''# Test oversold condition
        oversold_data = self.data.copy()
        oversold_data['close'] = 100 - np.abs(np.sin(np.arange(100)) * 20)
        signal = {0}(oversold_data)
        # Should identify oversold conditions'''.format(strategy_name)
    
    else:
        return f'''# Test general conditions
        signal = {strategy_name}(self.data)
        self.assertIsInstance(signal, Signal)'''

def create_test_infrastructure():
    """Create the necessary test infrastructure."""
    # Create test directory
    test_dir = PROJECT_ROOT / 'tests' / 'unit' / 'strategy' / 'indicators'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_file = test_dir / '__init__.py'
    init_file.write_text('"""Unit tests for indicator strategies."""\n')
    
    # Create conftest.py for pytest fixtures
    conftest_file = test_dir / 'conftest.py'
    conftest_content = '''"""
Pytest fixtures for indicator strategy tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    base_price = 100
    trend = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 2, 100)
    prices = base_price + trend + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

@pytest.fixture
def trending_data():
    """Create trending market data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 * (1.002 ** np.arange(100))  # 0.2% daily growth
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

@pytest.fixture
def ranging_data():
    """Create ranging market data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + 5 * np.sin(np.arange(100) * 0.1)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })
'''
    conftest_file.write_text(conftest_content)
    
    return test_dir

def main():
    """Main function to generate all indicator tests."""
    print("Creating test infrastructure...")
    test_dir = create_test_infrastructure()
    
    indicators_path = PROJECT_ROOT / 'src' / 'strategy' / 'strategies' / 'indicators'
    
    generated_tests = []
    
    for py_file in indicators_path.glob('*.py'):
        if py_file.name in ['__init__.py', 'crossovers_migrated.py']:
            continue
        # Skip temporary/hidden files
        if py_file.name.startswith('.'):
            continue
        
        module_name = py_file.stem
        print(f"\nProcessing {module_name}.py...")
        
        try:
            strategies = parse_strategy_functions(py_file)
            if strategies:
                print(f"  Found {len(strategies)} strategies")
                
                # Generate test content
                test_content = generate_test_content(module_name, strategies)
                
                # Write test file
                test_file = test_dir / f'test_{module_name}.py'
                test_file.write_text(test_content)
                
                generated_tests.append({
                    'module': module_name,
                    'strategies': len(strategies),
                    'test_file': str(test_file)
                })
                
                print(f"  Generated test file: {test_file}")
            else:
                print(f"  No strategies found")
                
        except Exception as e:
            print(f"  Error processing {module_name}: {e}")
    
    # Create summary
    print("\n" + "="*80)
    print("TEST GENERATION SUMMARY")
    print("="*80)
    print(f"Generated {len(generated_tests)} test files")
    
    for test in generated_tests:
        print(f"  - {test['module']}: {test['strategies']} strategies")
    
    # Create test automation script
    automation_script = test_dir / 'test_all_indicators.py'
    automation_content = '''#!/usr/bin/env python3
"""
Run all indicator strategy tests.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run all indicator tests."""
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob('test_*.py'))
    
    print(f"Running {len(test_files)} test files...")
    
    failed = []
    for test_file in test_files:
        if test_file.name == 'test_all_indicators.py':
            continue
            
        print(f"\\nRunning {test_file.name}...")
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            failed.append(test_file.name)
            print(f"  FAILED")
            print(result.stdout)
            print(result.stderr)
        else:
            print(f"  PASSED")
    
    if failed:
        print(f"\\n{len(failed)} tests failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\\nAll tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
'''
    automation_script.write_text(automation_content)
    automation_script.chmod(0o755)
    
    print(f"\nCreated test automation script: {automation_script}")
    print("\nTo run all indicator tests:")
    print(f"  python {automation_script}")

if __name__ == '__main__':
    main()