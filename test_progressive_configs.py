"""Test progressively complex YAML configurations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config.simple_validator import SimpleConfigValidator


def test_configurations():
    """Test all configuration files."""
    print("=" * 70)
    print("Testing Progressive YAML Configurations")
    print("=" * 70)
    
    configs = [
        ("Simple Backtest", "configs/simple_backtest.yaml"),
        ("Multi-Strategy Backtest", "configs/multi_strategy_backtest.yaml"),
        ("Parameter Optimization", "configs/parameter_optimization.yaml"),
        ("Walk-Forward Analysis", "configs/walk_forward_analysis.yaml")
    ]
    
    validator = SimpleConfigValidator()
    
    for name, config_path in configs:
        print(f"\n{name}:")
        print("-" * len(name))
        
        path = Path(config_path)
        if not path.exists():
            print(f"  ✗ File not found: {config_path}")
            continue
        
        result = validator.validate_file(path)
        
        if result.is_valid:
            print(f"  ✓ Valid configuration")
            
            # Show key details
            config = result.normalized_config
            print(f"  Type: {config.get('type')}")
            print(f"  Name: {config.get('name')}")
            
            if config.get('type') == 'backtest':
                print(f"  Strategies: {len(config.get('strategies', []))}")
                total_allocation = sum(s.get('allocation', 1.0) for s in config.get('strategies', []))
                print(f"  Total Allocation: {total_allocation:.1%}")
                
            elif config.get('type') == 'optimization':
                opt = config.get('optimization', {})
                print(f"  Method: {opt.get('method', 'unknown')}")
                if 'parameter_space' in opt:
                    print(f"  Parameters: {list(opt['parameter_space'].keys())}")
                if 'walk_forward' in opt:
                    print(f"  Walk-Forward: Yes")
                    wf = opt['walk_forward']
                    print(f"    - Optimization Window: {wf.get('optimization_window')} days")
                    print(f"    - Test Window: {wf.get('test_window')} days")
        else:
            print(f"  ✗ Invalid configuration")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        if result.warnings:
            print(f"  Warnings:")
            for warning in result.warnings:
                print(f"    ⚠ {warning}")
    
    print("\n" + "=" * 70)
    print("Configuration Complexity Progression:")
    print("=" * 70)
    
    print("\n1. Simple Backtest (Beginner)")
    print("   - Single strategy")
    print("   - Fixed position sizing")
    print("   - Basic risk limits")
    print("   - Quick to test ideas")
    
    print("\n2. Multi-Strategy Portfolio (Intermediate)")
    print("   - Multiple strategies with allocation")
    print("   - Risk parity position sizing")
    print("   - Advanced risk limits")
    print("   - Performance attribution")
    
    print("\n3. Parameter Optimization (Advanced)")
    print("   - Automatic parameter search")
    print("   - Multi-objective optimization")
    print("   - Parallel execution")
    print("   - Constraint handling")
    
    print("\n4. Walk-Forward Analysis (Expert)")
    print("   - Rolling window optimization")
    print("   - Out-of-sample validation")
    print("   - Parameter stability analysis")
    print("   - Robustness testing")
    
    print("\n" + "=" * 70)
    print("Key Benefits of Configuration-Driven Approach:")
    print("=" * 70)
    print("✓ No code to write or debug")
    print("✓ Identical execution paths for all workflows")
    print("✓ Complex analyses available to everyone")
    print("✓ Focus on strategy, not implementation")
    print("✓ Reproducible results")
    print("✓ Easy to share and version control")


def show_execution_commands():
    """Show how to execute each configuration."""
    print("\n\nExecution Commands:")
    print("-" * 40)
    
    commands = [
        ("Simple backtest", "python main.py --config configs/simple_backtest.yaml"),
        ("Multi-strategy", "python main.py --config configs/multi_strategy_backtest.yaml"),
        ("Optimization", "python main.py --config configs/parameter_optimization.yaml"),
        ("Walk-forward", "python main.py --config configs/walk_forward_analysis.yaml"),
        ("With debugging", "python main.py --config configs/simple_backtest.yaml --verbose"),
        ("Dry run only", "python main.py --config configs/optimization.yaml --dry-run")
    ]
    
    for desc, cmd in commands:
        print(f"\n{desc}:")
        print(f"  $ {cmd}")


if __name__ == "__main__":
    test_configurations()
    show_execution_commands()