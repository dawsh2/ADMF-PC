"""Simple demonstration of YAML interpretation and validation."""

import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import ConfigSchemaValidator
from src.core.coordinator.yaml_interpreter import YAMLInterpreter, YAMLWorkflowBuilder


def demo_yaml_interpretation():
    """Demonstrate YAML interpretation without full execution."""
    
    print("=" * 60)
    print("ADMF-PC YAML Configuration Demo")
    print("Zero-Code Trading System")
    print("=" * 60)
    
    # Create example YAML configuration
    backtest_yaml = """
name: Simple Moving Average Backtest
type: backtest
description: Demonstration of YAML-driven backtesting

data:
  symbols: ["SPY", "QQQ", "IWM"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  frequency: "1d"
  source: yahoo

portfolio:
  initial_capital: 100000
  currency: USD
  commission:
    type: fixed
    value: 1.0

strategies:
  - name: ma_crossover
    type: moving_average_crossover
    allocation: 0.5
    parameters:
      fast_period: 10
      slow_period: 30
      ma_type: SMA
    
  - name: momentum
    type: momentum_strategy
    allocation: 0.5
    parameters:
      lookback_period: 20
      entry_threshold: 0.02
      exit_threshold: -0.01

risk:
  position_sizers:
    - name: equal_weight
      type: percentage
      percentage: 33.33
      
  limits:
    - type: position
      max_position: 50000
    - type: exposure
      max_exposure_pct: 95
    - type: drawdown
      max_drawdown_pct: 20
      reduce_at_pct: 15

analysis:
  metrics: ["returns", "sharpe", "max_drawdown", "win_rate"]
  plots: ["equity_curve", "drawdown", "returns_distribution"]
"""

    # Save YAML to file
    yaml_dir = Path("examples/yaml_configs")
    yaml_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_file = yaml_dir / "demo_backtest.yaml"
    with open(yaml_file, "w") as f:
        f.write(backtest_yaml)
    
    print(f"\n1. Created YAML configuration: {yaml_file}")
    print("\n" + "-" * 40)
    print("Configuration Content:")
    print("-" * 40)
    print(backtest_yaml[:500] + "...")
    
    # Validate the configuration
    print("\n2. Validating configuration...")
    validator = ConfigSchemaValidator()
    validation_result = validator.validate_file(yaml_file)
    
    if validation_result.is_valid:
        print("   ✓ Configuration is valid!")
        if validation_result.warnings:
            print("   Warnings:")
            for warning in validation_result.warnings:
                print(f"     - {warning}")
    else:
        print("   ✗ Configuration has errors:")
        for error in validation_result.errors:
            print(f"     - {error}")
        return
    
    # Interpret the configuration
    print("\n3. Interpreting YAML configuration...")
    interpreter = YAMLInterpreter()
    
    try:
        workflow_config, _ = interpreter.load_and_interpret(yaml_file)
        
        print(f"\n   Workflow Type: {workflow_config.workflow_type.value}")
        print(f"   Name: {workflow_config.parameters.get('name', 'unnamed')}")
        print(f"\n   Data Configuration:")
        for key, value in workflow_config.data_config.items():
            print(f"     - {key}: {value}")
        
        print(f"\n   Strategies ({len(workflow_config.backtest_config['strategies'])}):")
        for strategy in workflow_config.backtest_config['strategies']:
            print(f"     - {strategy['name']} ({strategy['type']})")
            print(f"       Allocation: {strategy['allocation'] * 100}%")
            print(f"       Parameters: {strategy['parameters']}")
        
        print(f"\n   Risk Management:")
        risk_config = workflow_config.backtest_config['risk']
        print(f"     Position Sizers: {len(risk_config['position_sizers'])}")
        for sizer in risk_config['position_sizers']:
            print(f"       - {sizer['name']} ({sizer['type']})")
        print(f"     Risk Limits: {len(risk_config['risk_limits'])}")
        for limit in risk_config['risk_limits']:
            print(f"       - {limit['type']}: {limit['parameters']}")
            
    except Exception as e:
        print(f"   Error during interpretation: {str(e)}")
        return
    
    # Build container hierarchy
    print("\n4. Building Container Hierarchy...")
    builder = YAMLWorkflowBuilder(interpreter)
    container_spec = builder.build_container_hierarchy(workflow_config)
    
    def print_containers(spec, level=0):
        indent = "   " * level
        print(f"{indent}- {spec['type']} (ID: {spec['id']})")
        print(f"{indent}  Capabilities: {', '.join(spec.get('capabilities', []))}")
        if 'config' in spec and spec['config']:
            print(f"{indent}  Config keys: {list(spec['config'].keys())}")
        for child in spec.get('children', []):
            print_containers(child, level + 1)
    
    print("\n   Container Structure:")
    print_containers(container_spec['root'])
    
    # Show optimization example
    print("\n" + "=" * 60)
    print("Optimization Configuration Example")
    print("=" * 60)
    
    optimization_yaml = """
name: MA Strategy Optimization
type: optimization
description: Find optimal moving average parameters

base_config:
  data:
    symbols: ["SPY"]
    start_date: "2022-01-01"
    end_date: "2023-12-31"
    frequency: "1h"
  portfolio:
    initial_capital: 100000
  strategies:
    - name: ma_strategy
      type: moving_average_crossover

optimization:
  method: bayesian
  parameter_space:
    fast_period:
      type: int
      min: 5
      max: 50
    slow_period:
      type: int
      min: 20
      max: 200
  constraints:
    - type: expression
      expression: "slow_period > fast_period + 10"
  objectives:
    - metric: sharpe_ratio
      direction: maximize
      weight: 0.7
    - metric: max_drawdown
      direction: minimize
      weight: 0.3
  n_trials: 100
  n_jobs: 4

parallel: true
max_workers: 4
"""
    
    opt_file = yaml_dir / "demo_optimization.yaml"
    with open(opt_file, "w") as f:
        f.write(optimization_yaml)
    
    # Validate and interpret
    print("\nValidating optimization configuration...")
    opt_result = validator.validate_file(opt_file)
    
    if opt_result.is_valid:
        print("✓ Optimization config is valid!")
        
        opt_config, _ = interpreter.load_and_interpret(opt_file)
        opt_spec = opt_config.optimization_config
        
        print(f"\nOptimization Details:")
        print(f"  Method: {opt_spec['method']}")
        print(f"  Trials: {opt_spec['n_trials']}")
        print(f"  Parallel Jobs: {opt_spec['n_jobs']}")
        print(f"  Parameter Space:")
        for param, spec in opt_spec['parameter_space'].items():
            print(f"    - {param}: {spec}")
        print(f"  Objectives:")
        for obj in opt_spec['objectives']:
            print(f"    - {obj['metric']} ({obj['direction']}) - weight: {obj.get('weight', 1.0)}")
    
    # Show validation examples
    print("\n" + "=" * 60)
    print("Configuration Validation Examples")
    print("=" * 60)
    
    # Invalid config
    invalid_yaml = """
name: Invalid Config
type: backtest
# Missing required fields!
"""
    
    invalid_file = yaml_dir / "invalid_config.yaml"
    with open(invalid_file, "w") as f:
        f.write(invalid_yaml)
    
    print("\n1. Invalid Configuration:")
    invalid_result = validator.validate_file(invalid_file)
    print(f"   Valid: {invalid_result.is_valid}")
    for error in invalid_result.errors[:3]:  # Show first 3 errors
        print(f"   ✗ {error}")
    
    # Config with warnings
    warning_yaml = """
name: Config with Warnings
type: live_trading
paper_trading: false  # Live trading!

broker:
  name: alpaca
  # Missing API key for live trading!

data:
  provider: alpaca
  symbols: ["SPY"]
  frequency: "1min"

portfolio:
  initial_capital: 10000

strategies:
  - name: risky_strategy
    type: high_frequency

risk:
  limits:
    - type: position
      max_position: 5000
    # No daily loss limit!
"""
    
    warning_file = yaml_dir / "warning_config.yaml"
    with open(warning_file, "w") as f:
        f.write(warning_yaml)
    
    print("\n2. Configuration with Warnings:")
    warning_result = validator.validate_file(warning_file)
    print(f"   Valid: {warning_result.is_valid}")
    print(f"   Errors: {len(warning_result.errors)}")
    for error in warning_result.errors:
        print(f"   ✗ {error}")
    print(f"   Warnings: {len(warning_result.warnings)}")
    for warning in warning_result.warnings:
        print(f"   ⚠ {warning}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nADMF-PC enables zero-code trading through:")
    print("1. YAML-based configuration files")
    print("2. Automatic validation before execution")
    print("3. Intelligent interpretation into execution plans")
    print("4. Container-based architecture for isolation")
    print("5. Support for backtesting, optimization, and live trading")
    
    print("\nNext steps would be to implement:")
    print("- Data loading components")
    print("- Strategy execution engines")
    print("- Risk management systems")
    print("- Performance analytics")
    print("- CLI interface for easy execution")


if __name__ == "__main__":
    demo_yaml_interpretation()