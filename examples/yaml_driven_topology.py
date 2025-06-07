"""
Example of using the YAML-driven topology system.

This shows how users can run complete workflows using only YAML configuration,
without writing any Python code for topology creation.
"""

import yaml
from pathlib import Path
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator import Coordinator


def run_yaml_backtest():
    """Run a backtest using pure YAML configuration."""
    
    # User configuration (could also come from YAML)
    user_config = {
        'symbols': ['SPY', 'QQQ'],
        'timeframes': ['5T', '15T'],
        'data_source': 'file',
        'data_path': './data',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        
        'strategies': [
            {'type': 'momentum', 'fast_period': 10, 'slow_period': 30},
            {'type': 'mean_reversion', 'lookback': 20, 'num_std': 2}
        ],
        
        'risk_profiles': [
            {'type': 'conservative', 'max_position_size': 0.1},
            {'type': 'moderate', 'max_position_size': 0.2}
        ],
        
        'features': {
            'returns': {'periods': [1, 5, 20]},
            'volume': {'ma_period': 20},
            'volatility': {'period': 20}
        },
        
        'initial_capital': 100000,
        'execution': {
            'commission': 0.001,
            'slippage': 0.0005
        }
    }
    
    # Create topology builder
    builder = TopologyBuilder()
    
    # Build topology from pattern and config
    topology = builder.build_topology({
        'mode': 'backtest',
        'config': user_config,
        'tracing_config': {
            'enabled': True,
            'trace_id': 'example_001'
        }
    })
    
    # Create and run coordinator
    coordinator = Coordinator()
    coordinator.run_workflow({
        'topology': topology,
        'sequence': 'single_pass'  # or 'walk_forward', 'monte_carlo', etc.
    })
    
    print(f"Backtest completed with {len(topology['containers'])} containers")


def run_signal_generation_and_replay():
    """Generate signals first, then replay them."""
    
    # Phase 1: Generate signals
    signal_gen_config = {
        'symbols': ['SPY'],
        'timeframes': ['5T'],
        'data_source': 'file',
        'data_path': './data',
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        
        'strategies': [
            {'type': 'momentum', 'fast_period': 10, 'slow_period': 30}
        ],
        
        'signal_save_directory': './results/signals/experiment_001/'
    }
    
    builder = TopologyBuilder()
    
    # Generate signals
    signal_topology = builder.build_topology({
        'mode': 'signal_generation',
        'config': signal_gen_config
    })
    
    coordinator = Coordinator()
    coordinator.run_workflow({
        'topology': signal_topology,
        'sequence': 'single_pass'
    })
    
    print("Signal generation completed")
    
    # Phase 2: Replay signals with different risk profiles
    replay_config = {
        'signal_directory': './results/signals/experiment_001/',
        'symbol': 'SPY',
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        
        'strategies': [
            {'type': 'momentum'}  # Must match what was used in generation
        ],
        
        'risk_profiles': [
            {'type': 'conservative', 'max_position_size': 0.1},
            {'type': 'moderate', 'max_position_size': 0.2},
            {'type': 'aggressive', 'max_position_size': 0.3}
        ],
        
        'initial_capital': 100000
    }
    
    # Replay signals
    replay_topology = builder.build_topology({
        'mode': 'signal_replay',
        'config': replay_config
    })
    
    coordinator.run_workflow({
        'topology': replay_topology,
        'sequence': 'single_pass'
    })
    
    print(f"Signal replay completed with {len(replay_config['risk_profiles'])} risk profiles")


def load_and_run_custom_pattern():
    """Load a custom pattern from YAML and run it."""
    
    # User creates their own pattern file
    custom_pattern_path = Path('./my_patterns/custom_backtest.yaml')
    
    # Load the pattern
    with open(custom_pattern_path) as f:
        custom_pattern = yaml.safe_load(f)
    
    # Register it with the builder
    builder = TopologyBuilder()
    builder.patterns['custom_backtest'] = custom_pattern
    
    # Use it like any other pattern
    topology = builder.build_topology({
        'mode': 'custom_backtest',
        'config': {
            # User configuration here
        }
    })
    
    print(f"Running custom pattern: {custom_pattern['name']}")


def inspect_available_patterns():
    """Show what patterns are available."""
    
    builder = TopologyBuilder()
    
    print("Available topology patterns:")
    for mode in builder.get_supported_modes():
        pattern = builder.get_pattern(mode)
        if pattern:
            print(f"\n{mode}:")
            print(f"  Description: {pattern.get('description', 'No description')}")
            print(f"  Containers: {len(pattern.get('containers', []))}")
            print(f"  Routes: {len(pattern.get('routes', []))}")
            print(f"  Behaviors: {len(pattern.get('behaviors', []))}")


if __name__ == "__main__":
    # Example 1: Simple backtest
    print("=== Running YAML-driven backtest ===")
    run_yaml_backtest()
    
    # Example 2: Signal generation and replay
    print("\n=== Running signal generation and replay ===")
    run_signal_generation_and_replay()
    
    # Example 3: Show available patterns
    print("\n=== Available patterns ===")
    inspect_available_patterns()