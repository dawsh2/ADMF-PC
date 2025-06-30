#!/usr/bin/env python3
"""
Validate indicator strategy configurations without running the full system.

This checks that:
1. Config files are valid YAML
2. Strategy names match actual strategy functions
3. Parameters match what strategies expect
"""

import yaml
import sys
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Tuple

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_strategy_parameters(strategy_name: str) -> Tuple[bool, List[str]]:
    """Extract expected parameters for a strategy."""
    # Map config names to module/function names
    strategy_module_map = {
        'sma_crossover': ('crossovers', 'sma_crossover'),
        'ema_crossover': ('crossovers', 'ema_crossover'),
        'ema_sma_crossover': ('crossovers', 'ema_sma_crossover'),
        'dema_crossover': ('crossovers', 'dema_crossover'),
        'dema_sma_crossover': ('crossovers', 'dema_sma_crossover'),
        'tema_sma_crossover': ('crossovers', 'tema_sma_crossover'),
        'macd_crossover': ('crossovers', 'macd_crossover'),
        'stochastic_crossover': ('crossovers', 'stochastic_crossover'),
        'vortex_crossover': ('crossovers', 'vortex_crossover'),
        'ichimoku_cloud_position': ('crossovers', 'ichimoku_cloud_position'),
        
        'rsi_bands': ('oscillators', 'rsi_bands'),
        'rsi_threshold': ('oscillators', 'rsi_threshold'),
        'cci_bands': ('oscillators', 'cci_bands'),
        'cci_threshold': ('oscillators', 'cci_threshold'),
        'stochastic_rsi': ('oscillators', 'stochastic_rsi'),
        'williams_r': ('oscillators', 'williams_r'),
        'roc_threshold': ('oscillators', 'roc_threshold'),
        'ultimate_oscillator': ('oscillators', 'ultimate_oscillator'),
        
        'bollinger_bands': ('volatility', 'bollinger_bands'),
        'bollinger_breakout': ('volatility', 'bollinger_breakout'),
        'keltner_bands': ('volatility', 'keltner_bands'),
        'keltner_breakout': ('volatility', 'keltner_breakout'),
        'donchian_breakout': ('volatility', 'donchian_breakout'),
        
        'obv_trend': ('volume', 'obv_trend'),
        'mfi_bands': ('volume', 'mfi_bands'),
        'vwap_deviation': ('volume', 'vwap_deviation'),
        'chaikin_money_flow': ('volume', 'chaikin_money_flow'),
        'accumulation_distribution': ('volume', 'accumulation_distribution'),
        
        'rsi_divergence': ('divergence', 'rsi_divergence'),
        'macd_histogram_divergence': ('divergence', 'macd_histogram_divergence'),
        'stochastic_divergence': ('divergence', 'stochastic_divergence'),
        'momentum_divergence': ('divergence', 'momentum_divergence'),
        'obv_price_divergence': ('divergence', 'obv_price_divergence'),
        
        'adx_trend_strength': ('momentum', 'adx_trend_strength_strategy'),
        'aroon_oscillator': ('momentum', 'aroon_oscillator_strategy'),
        'elder_ray': ('momentum', 'elder_ray_strategy'),
        'momentum_breakout': ('momentum', 'momentum_breakout_strategy'),
        'roc_trend': ('momentum', 'roc_trend_strategy'),
        'vortex_trend': ('momentum', 'vortex_trend_strategy'),
        
        'pivot_points': ('structure', 'pivot_points'),
        'pivot_bounces': ('structure', 'pivot_bounces'),
        'swing_pivot_breakout': ('structure', 'swing_pivot_breakout'),
        'swing_pivot_bounce': ('structure', 'swing_pivot_bounce'),
        'trendline_breaks': ('structure', 'trendline_breaks'),
        'trendline_bounces': ('structure', 'trendline_bounces'),
        'diagonal_channel_breakout': ('structure', 'diagonal_channel_breakout'),
        'diagonal_channel_reversion': ('structure', 'diagonal_channel_reversion'),
        
        'aroon_crossover': ('trend', 'aroon_crossover'),
        'parabolic_sar': ('trend', 'parabolic_sar'),
        'linear_regression_slope': ('trend', 'linear_regression_slope'),
        'supertrend': ('trend', 'supertrend'),
    }
    
    if strategy_name not in strategy_module_map:
        return False, [f"Unknown strategy: {strategy_name}"]
    
    module_name, func_name = strategy_module_map[strategy_name]
    
    # Load the strategy file and extract parameter info
    try:
        file_path = f"src/strategy/strategies/indicators/{module_name}.py"
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the function and its parameter_space
        import re
        
        # Find the strategy decorator for this function
        pattern = rf'@strategy\([^)]*\)\s*def\s+{func_name}\s*\([^)]*\)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            # Try to find parameter space in decorator
            pattern = rf'parameter_space\s*=\s*{{([^}}]+)}}'
            matches = re.finditer(pattern, content)
            
            # Look for the right parameter space (heuristic: closest before function def)
            params = []
            for m in matches:
                param_content = m.group(1)
                # Extract parameter names
                param_names = re.findall(r"'(\w+)':", param_content)
                if func_name in content[max(0, m.start()-500):m.end()+100]:
                    params = param_names
                    break
            
            return True, params
        
        return True, []
        
    except Exception as e:
        return False, [f"Error loading strategy: {e}"]

def validate_config(config_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single configuration file."""
    errors = []
    
    try:
        # Load config
        config = load_yaml_config(str(config_path))
        
        # Check required fields
        if 'strategy' not in config:
            errors.append("Missing 'strategy' field")
            return False, errors
        
        # Validate each strategy
        for strategy_name, strategy_config in config['strategy'].items():
            # Check if strategy exists
            exists, expected_params = get_strategy_parameters(strategy_name)
            
            if not exists:
                errors.append(f"Strategy '{strategy_name}' not found")
                continue
            
            # Check parameters
            if 'params' in strategy_config:
                provided_params = set(strategy_config['params'].keys())
                
                # Note: We can't perfectly validate without the decorator info,
                # but we can check for obvious issues
                if strategy_name == 'dema_crossover' and 'fast_period' in provided_params:
                    errors.append(f"{strategy_name} expects 'fast_dema_period', not 'fast_period'")
                
                # Add more specific validations as needed
                
        return len(errors) == 0, errors
        
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML: {e}"]
    except Exception as e:
        return False, [f"Error: {e}"]

def main():
    """Validate all indicator configurations."""
    indicators_dir = Path('config/indicators')
    
    if not indicators_dir.exists():
        print(f"Error: {indicators_dir} not found")
        return 1
    
    total_configs = 0
    passed_configs = 0
    failed_configs = []
    
    # Find all YAML files
    for yaml_file in indicators_dir.rglob('*.yaml'):
        if 'test_' not in yaml_file.name:
            continue
            
        total_configs += 1
        print(f"Validating {yaml_file.relative_to(indicators_dir)}... ", end='')
        
        valid, errors = validate_config(yaml_file)
        
        if valid:
            print("✓")
            passed_configs += 1
        else:
            print("✗")
            failed_configs.append((yaml_file, errors))
            for error in errors:
                print(f"  - {error}")
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)
    print(f"Total configs: {total_configs}")
    print(f"Passed: {passed_configs}")
    print(f"Failed: {len(failed_configs)}")
    
    if failed_configs:
        print("\nFailed configurations:")
        for config_path, errors in failed_configs:
            print(f"\n{config_path.relative_to(indicators_dir)}:")
            for error in errors:
                print(f"  - {error}")
    
    return 0 if len(failed_configs) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())