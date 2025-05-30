"""Demonstration of YAML-driven workflow execution in ADMF-PC.

This example shows how users can define and execute trading workflows
using only YAML configurations - no coding required!
"""

import asyncio
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.coordinator.yaml_coordinator import YAMLCoordinator, run_backtest
from src.core.config import ConfigSchemaValidator


async def demo_yaml_workflow():
    """Demonstrate YAML-driven workflow execution."""
    
    print("=" * 60)
    print("ADMF-PC YAML Workflow Demonstration")
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
    
    # Create YAML Coordinator
    print("\n3. Creating YAML Coordinator...")
    coordinator = YAMLCoordinator()
    
    # Execute the workflow
    print("\n4. Executing YAML workflow...")
    print("   This would normally:")
    print("   - Load historical data for SPY, QQQ, IWM")
    print("   - Run two strategies with 50% allocation each")
    print("   - Apply risk limits and position sizing")
    print("   - Generate performance analytics")
    
    # In a real implementation, this would execute the backtest
    # For now, we'll demonstrate the interpretation
    workflow_config, _ = coordinator.yaml_interpreter.load_and_interpret(yaml_file)
    
    print("\n5. Interpreted Configuration:")
    print(f"   Workflow Type: {workflow_config.workflow_type.value}")
    print(f"   Name: {workflow_config.parameters['name']}")
    print(f"   Data Config: {workflow_config.data_config}")
    print(f"   Strategies: {len(workflow_config.backtest_config['strategies'])}")
    print(f"   Risk Limits: {len(workflow_config.backtest_config['risk']['risk_limits'])}")
    
    # Build container hierarchy
    print("\n6. Container Hierarchy:")
    container_spec = coordinator.workflow_builder.build_container_hierarchy(workflow_config)
    
    def print_containers(spec, level=0):
        indent = "   " * level
        print(f"{indent}- {spec['type']} ({spec['id']})")
        print(f"{indent}  Capabilities: {', '.join(spec['capabilities'])}")
        for child in spec.get('children', []):
            print_containers(child, level + 1)
    
    print_containers(container_spec['root'])
    
    # Demonstrate other workflow types
    print("\n" + "=" * 60)
    print("Other Workflow Types Supported:")
    print("=" * 60)
    
    # Optimization example
    optimization_yaml = """
name: Strategy Parameter Optimization
type: optimization

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
  n_trials: 100
"""
    
    print("\n1. Optimization Workflow:")
    print("   - Automatically searches parameter space")
    print("   - Runs multiple backtests in parallel")
    print("   - Finds optimal parameters based on objectives")
    
    # Live trading example
    live_yaml = """
name: Live Momentum Trading
type: live_trading
paper_trading: true

broker:
  name: paper

data:
  provider: yahoo
  symbols: ["SPY", "QQQ"]
  frequency: "5min"
  lookback_days: 30

portfolio:
  initial_capital: 100000

strategies:
  - name: momentum
    type: momentum_strategy
    parameters:
      lookback_period: 20

risk:
  limits:
    - type: daily_loss
      max_daily_loss_pct: 2.0
"""
    
    print("\n2. Live Trading Workflow:")
    print("   - Connects to real-time data feeds")
    print("   - Executes strategies in real-time")
    print("   - Monitors risk limits continuously")
    print("   - Can switch between paper and live trading")
    
    print("\n" + "=" * 60)
    print("Key Benefits of YAML-Driven Approach:")
    print("=" * 60)
    print("1. No coding required - accessible to all traders")
    print("2. Configuration validation catches errors early")
    print("3. Standardized format across all workflow types")
    print("4. Easy to version control and share")
    print("5. Can be generated by UI tools")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Implement actual data loading")
    print("2. Build strategy execution engine")
    print("3. Add performance analytics")
    print("4. Create CLI for easy execution:")
    print("   $ admf-pc backtest my_strategy.yaml")
    print("   $ admf-pc optimize optimization_config.yaml")
    print("   $ admf-pc live live_trading.yaml")


async def demo_validation():
    """Demonstrate configuration validation."""
    print("\n" + "=" * 60)
    print("Configuration Validation Demo")
    print("=" * 60)
    
    # Invalid configuration
    invalid_yaml = """
name: Invalid Config
type: backtest

# Missing required fields: data, portfolio, strategies
"""
    
    yaml_file = Path("examples/yaml_configs/invalid.yaml")
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_file, "w") as f:
        f.write(invalid_yaml)
    
    coordinator = YAMLCoordinator()
    validation_result = await coordinator.validate_yaml(yaml_file)
    
    print(f"\nValidating {yaml_file}:")
    print(f"Valid: {validation_result.is_valid}")
    print("Errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
    
    # Configuration with warnings
    warning_yaml = """
name: Config with Warnings
type: backtest

data:
  symbols: ["AAPL"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  frequency: "1d"

portfolio:
  initial_capital: 100000

strategies:
  - name: strategy1
    type: test_strategy
    allocation: 0.6
  - name: strategy2
    type: test_strategy2
    allocation: 0.3
    # Total allocation is 0.9, not 1.0
"""
    
    yaml_file2 = Path("examples/yaml_configs/warning.yaml") 
    with open(yaml_file2, "w") as f:
        f.write(warning_yaml)
    
    validation_result2 = await coordinator.validate_yaml(yaml_file2)
    
    print(f"\nValidating {yaml_file2}:")
    print(f"Valid: {validation_result2.is_valid}")
    if validation_result2.warnings:
        print("Warnings:")
        for warning in validation_result2.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_yaml_workflow())
    asyncio.run(demo_validation())