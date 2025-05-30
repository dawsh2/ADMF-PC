"""Standalone YAML demonstration without Coordinator dependencies."""

import yaml
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only what we need
from src.core.config.simple_validator import SimpleConfigValidator


def interpret_yaml_config(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Simple interpretation of YAML config."""
    # Detect workflow type
    config_type = yaml_config.get("type", "backtest")
    
    # Build interpreted config
    interpreted = {
        "workflow_type": config_type,
        "name": yaml_config.get("name", f"{config_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "description": yaml_config.get("description", ""),
        "data_config": yaml_config.get("data", {}),
        "strategies": [],
        "risk_config": {},
        "container_hierarchy": {}
    }
    
    # Interpret strategies
    for strategy in yaml_config.get("strategies", []):
        interpreted["strategies"].append({
            "name": strategy["name"],
            "type": strategy["type"],
            "allocation": strategy.get("allocation", 1.0),
            "parameters": strategy.get("parameters", {}),
            "enabled": strategy.get("enabled", True)
        })
    
    # Interpret risk config
    if "risk" in yaml_config:
        risk = yaml_config["risk"]
        interpreted["risk_config"] = {
            "position_sizers": risk.get("position_sizers", []),
            "limits": risk.get("limits", [])
        }
    
    # Build container hierarchy
    interpreted["container_hierarchy"] = build_container_spec(yaml_config, config_type)
    
    return interpreted


def build_container_spec(yaml_config: Dict[str, Any], workflow_type: str) -> Dict[str, Any]:
    """Build container hierarchy specification."""
    spec = {
        "root": {
            "type": "workflow",
            "id": f"{workflow_type}_root",
            "capabilities": ["monitoring", "error_handling", "event_bus"],
            "children": []
        }
    }
    
    # Add data container
    if "data" in yaml_config:
        spec["root"]["children"].append({
            "type": "data",
            "id": "data_container",
            "capabilities": ["data_loading", "data_validation"],
            "config": yaml_config["data"]
        })
    
    # Add strategy containers
    for i, strategy in enumerate(yaml_config.get("strategies", [])):
        spec["root"]["children"].append({
            "type": "strategy",
            "id": f"strategy_{strategy['name']}",
            "capabilities": ["backtesting", "signal_generation"],
            "config": strategy
        })
    
    # Add risk container
    if "risk" in yaml_config:
        spec["root"]["children"].append({
            "type": "risk_portfolio",
            "id": "risk_portfolio_container",
            "capabilities": ["risk_management", "position_sizing"],
            "config": yaml_config["risk"]
        })
    
    return spec


def demo_yaml_system():
    """Demonstrate the YAML-driven system."""
    
    print("=" * 70)
    print("ADMF-PC: YAML-Driven Trading System Demonstration")
    print("=" * 70)
    
    # Example 1: Simple Backtest
    print("\n1. BACKTEST CONFIGURATION")
    print("-" * 40)
    
    backtest_yaml = """
name: Moving Average Crossover Backtest
type: backtest
description: Test a simple MA crossover strategy on tech stocks

data:
  symbols: ["AAPL", "GOOGL", "MSFT"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  frequency: "1d"
  source: yahoo

portfolio:
  initial_capital: 100000
  currency: USD
  commission:
    type: percentage
    value: 0.001

strategies:
  - name: ma_crossover
    type: moving_average_crossover
    allocation: 1.0
    parameters:
      fast_period: 20
      slow_period: 50
      ma_type: EMA

risk:
  position_sizers:
    - name: equal_weight
      type: percentage
      percentage: 33.33
      
  limits:
    - type: position
      max_position: 50000
    - type: drawdown
      max_drawdown_pct: 15
      reduce_at_pct: 10

analysis:
  metrics: ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
  export_results: true
"""
    
    # Save and validate
    yaml_dir = Path("examples/yaml_configs")
    yaml_dir.mkdir(parents=True, exist_ok=True)
    
    backtest_file = yaml_dir / "backtest_demo.yaml"
    with open(backtest_file, "w") as f:
        f.write(backtest_yaml)
    
    print(f"Created: {backtest_file}")
    
    # Validate
    validator = SimpleConfigValidator()
    result = validator.validate_file(backtest_file)
    
    print(f"\nValidation: {'✓ PASSED' if result.is_valid else '✗ FAILED'}")
    if result.warnings:
        for warning in result.warnings:
            print(f"  Warning: {warning}")
    
    # Interpret
    yaml_dict = yaml.safe_load(backtest_yaml)
    interpreted = interpret_yaml_config(yaml_dict)
    
    print(f"\nInterpreted Configuration:")
    print(f"  Workflow: {interpreted['workflow_type']}")
    print(f"  Strategies: {len(interpreted['strategies'])}")
    for strategy in interpreted['strategies']:
        print(f"    - {strategy['name']} ({strategy['allocation']*100:.0f}% allocation)")
    print(f"  Risk Limits: {len(interpreted['risk_config']['limits'])}")
    
    # Show container structure
    print(f"\nContainer Architecture:")
    def print_containers(node, level=0):
        indent = "  " * level
        print(f"{indent}├─ {node['type']} [{node['id']}]")
        print(f"{indent}│  Capabilities: {', '.join(node['capabilities'])}")
        for child in node.get('children', []):
            print_containers(child, level + 1)
    
    print_containers(interpreted['container_hierarchy']['root'])
    
    # Example 2: Optimization
    print("\n\n2. OPTIMIZATION CONFIGURATION")
    print("-" * 40)
    
    optimization_yaml = """
name: Parameter Optimization
type: optimization
description: Find optimal MA periods for maximum Sharpe ratio

base_config:
  data:
    symbols: ["SPY"]
    start_date: "2020-01-01"
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
      step: 5
    slow_period:
      type: int
      min: 20
      max: 200
      step: 10
    ma_type:
      type: choice
      choices: ["SMA", "EMA", "WMA"]
  
  constraints:
    - type: expression
      expression: "slow_period > fast_period + 10"
  
  objectives:
    - metric: sharpe_ratio
      direction: maximize
      weight: 0.6
    - metric: calmar_ratio
      direction: maximize
      weight: 0.4
  
  n_trials: 200
  n_jobs: 4
  early_stopping:
    enabled: true
    patience: 50

parallel: true
max_workers: 4
"""
    
    opt_file = yaml_dir / "optimization_demo.yaml"
    with open(opt_file, "w") as f:
        f.write(optimization_yaml)
    
    print(f"Created: {opt_file}")
    
    # Validate
    opt_result = validator.validate_file(opt_file)
    print(f"\nValidation: {'✓ PASSED' if opt_result.is_valid else '✗ FAILED'}")
    
    # Example 3: Live Trading
    print("\n\n3. LIVE TRADING CONFIGURATION")
    print("-" * 40)
    
    live_yaml = """
name: Momentum Live Trading
type: live_trading
description: Trade momentum strategy on liquid ETFs

paper_trading: true  # Start with paper trading

broker:
  name: paper
  timeout: 30
  retry_count: 3

data:
  provider: yahoo
  symbols: ["SPY", "QQQ", "IWM", "DIA"]
  frequency: "5min"
  lookback_days: 30
  warmup_period: 100

portfolio:
  initial_capital: 50000
  max_positions: 2
  position_sizing: risk_parity

strategies:
  - name: momentum
    type: momentum_strategy
    parameters:
      lookback_period: 20
      entry_percentile: 80
      exit_percentile: 50
      use_volume_filter: true

risk:
  check_frequency: pre_trade
  
  position_sizers:
    - name: volatility_adjusted
      type: volatility
      risk_per_trade: 1.0
      lookback_period: 20
  
  limits:
    - type: position
      max_position: 25000
    - type: exposure
      max_exposure_pct: 80
    - type: daily_loss
      max_daily_loss_pct: 2.0
    - type: concentration
      max_position_pct: 50
  
  emergency_shutdown:
    enabled: true
    max_daily_loss_pct: 3.0
    max_drawdown_pct: 10.0

execution:
  order_type: limit
  time_in_force: DAY
  split_orders:
    enabled: true
    max_order_size: 5000

monitoring:
  heartbeat_interval: 60
  log_level: INFO
  alerts:
    webhook: "https://hooks.example.com/trading-alerts"
"""
    
    live_file = yaml_dir / "live_trading_demo.yaml"
    with open(live_file, "w") as f:
        f.write(live_yaml)
    
    print(f"Created: {live_file}")
    
    # Validate
    live_result = validator.validate_file(live_file)
    print(f"\nValidation: {'✓ PASSED' if live_result.is_valid else '✗ FAILED'}")
    if live_result.warnings:
        for warning in live_result.warnings:
            print(f"  Warning: {warning}")
    
    # Show key features
    print("\n" + "=" * 70)
    print("KEY FEATURES OF YAML-DRIVEN APPROACH")
    print("=" * 70)
    
    features = [
        ("Zero-Code Trading", "Define complete trading systems without writing code"),
        ("Validation", "Catch configuration errors before execution"),
        ("Type Safety", "Ensure parameters match expected types and ranges"),
        ("Modular Design", "Mix and match strategies, risk limits, and data sources"),
        ("Version Control", "Track strategy evolution with standard Git workflows"),
        ("Shareable", "Share strategies as simple YAML files"),
        ("UI-Friendly", "Easy to generate from graphical interfaces"),
        ("Extensible", "Add new strategies without changing core system")
    ]
    
    for feature, description in features:
        print(f"\n{feature}:")
        print(f"  {description}")
    
    # Show workflow
    print("\n" + "=" * 70)
    print("TYPICAL WORKFLOW")
    print("=" * 70)
    
    workflow_steps = [
        "Create YAML configuration file",
        "Validate configuration",
        "Run backtest to test strategy",
        "Optimize parameters if needed",
        "Paper trade to verify live performance",
        "Switch to live trading when ready"
    ]
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"\n{i}. {step}")
        if i == 1:
            print("   $ vim strategy.yaml")
        elif i == 2:
            print("   $ admf-pc validate strategy.yaml")
        elif i == 3:
            print("   $ admf-pc backtest strategy.yaml")
        elif i == 4:
            print("   $ admf-pc optimize optimization.yaml")
        elif i == 5:
            print("   $ admf-pc live --paper trading.yaml")
        elif i == 6:
            print("   $ admf-pc live --real trading.yaml")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nADMF-PC transforms trading strategy development by:")
    print("• Eliminating coding barriers")
    print("• Providing enterprise-grade risk management")
    print("• Supporting the complete trading lifecycle")
    print("• Enabling rapid strategy iteration")
    print("• Ensuring reproducible results")
    
    print("\nAll through simple, validated YAML configurations!")


if __name__ == "__main__":
    demo_yaml_system()