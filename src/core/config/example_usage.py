"""Example usage of the configuration schema validator."""

import yaml
from pathlib import Path
from .schema_validator import SimpleConfigValidator as ConfigSchemaValidator
from .schemas import (
    BACKTEST_SCHEMA,
    OPTIMIZATION_SCHEMA,
    LIVE_TRADING_SCHEMA
)


def validate_yaml_file(file_path: str):
    """Example: Validate a YAML configuration file."""
    # Create validator
    validator = ConfigSchemaValidator()
    
    # Register schemas
    validator.register_schema(BACKTEST_SCHEMA)
    validator.register_schema(OPTIMIZATION_SCHEMA)
    validator.register_schema(LIVE_TRADING_SCHEMA)
    
    # Validate file
    result = validator.validate_file(file_path)
    
    if result.is_valid:
        print(f"✓ Configuration is valid!")
        print(f"Normalized config: {result.normalized_config}")
    else:
        print(f"✗ Configuration is invalid!")
        print(f"Errors:")
        for error in result.errors:
            print(f"  - {error}")
        if result.warnings:
            print(f"Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")


def validate_config_dict():
    """Example: Validate a configuration dictionary."""
    # Create validator
    validator = ConfigSchemaValidator()
    validator.register_schema(BACKTEST_SCHEMA)
    
    # Example configuration
    config = {
        "name": "Example Backtest",
        "data": {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 100000
        },
        "strategies": [{
            "name": "ma_crossover",
            "type": "moving_average_crossover",
            "parameters": {
                "fast_period": 10,
                "slow_period": 30
            }
        }],
        "risk": {
            "limits": [{
                "type": "position",
                "max_position": 10000
            }, {
                "type": "drawdown",
                "max_drawdown_pct": 20
            }]
        }
    }
    
    # Validate
    result = validator.validate(config, "backtest")
    
    if result.is_valid:
        print("✓ Configuration is valid!")
    else:
        print("✗ Configuration has errors:")
        for error in result.errors:
            print(f"  - {error}")


def create_sample_configs():
    """Create sample configuration files."""
    samples_dir = Path("configs/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample backtest config
    backtest_config = {
        "name": "Simple MA Crossover Backtest",
        "type": "backtest",
        "description": "Backtest of moving average crossover strategy",
        "data": {
            "symbols": ["SPY", "QQQ", "IWM"],
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d",
            "source": "yahoo"
        },
        "portfolio": {
            "initial_capital": 100000,
            "commission": {
                "type": "fixed",
                "value": 1.0
            },
            "slippage": {
                "type": "percentage",
                "value": 0.1
            }
        },
        "strategies": [{
            "name": "ma_cross",
            "type": "moving_average_crossover",
            "allocation": 1.0,
            "parameters": {
                "fast_period": 20,
                "slow_period": 50,
                "ma_type": "SMA"
            }
        }],
        "risk": {
            "position_sizers": [{
                "name": "equal_weight",
                "type": "percentage",
                "percentage": 33.33
            }],
            "limits": [{
                "type": "position",
                "max_position": 50000
            }, {
                "type": "exposure",
                "max_exposure_pct": 95
            }, {
                "type": "drawdown",
                "max_drawdown_pct": 20,
                "reduce_at_pct": 15
            }]
        },
        "analysis": {
            "metrics": ["returns", "sharpe", "max_drawdown", "win_rate"],
            "plots": ["equity_curve", "drawdown", "returns_distribution"],
            "export": {
                "format": "html",
                "path": "results/backtest_report.html"
            }
        }
    }
    
    with open(samples_dir / "backtest_example.yaml", "w") as f:
        yaml.dump(backtest_config, f, default_flow_style=False)
    
    # Sample optimization config  
    optimization_config = {
        "name": "MA Strategy Parameter Optimization",
        "type": "optimization",
        "description": "Optimize MA crossover parameters for best Sharpe ratio",
        "base_config": {
            "data": {
                "symbols": ["SPY"],
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "frequency": "1h"
            },
            "portfolio": {
                "initial_capital": 100000
            },
            "strategies": [{
                "name": "ma_cross",
                "type": "moving_average_crossover"
            }]
        },
        "optimization": {
            "method": "bayesian",
            "parameter_space": {
                "fast_period": {
                    "type": "int",
                    "min": 5,
                    "max": 50,
                    "step": 5
                },
                "slow_period": {
                    "type": "int",
                    "min": 20,
                    "max": 200,
                    "step": 10
                },
                "ma_type": {
                    "type": "choice",
                    "choices": ["SMA", "EMA", "WMA"]
                }
            },
            "constraints": [{
                "type": "expression",
                "expression": "slow_period > fast_period + 10"
            }],
            "objectives": [{
                "metric": "sharpe_ratio",
                "direction": "maximize",
                "weight": 0.6
            }, {
                "metric": "max_drawdown",
                "direction": "minimize", 
                "weight": 0.4
            }],
            "n_trials": 100,
            "n_jobs": 4,
            "early_stopping": {
                "enabled": True,
                "patience": 20,
                "min_delta": 0.01
            }
        },
        "parallel": True,
        "max_workers": 4,
        "output": {
            "save_top_n": 10,
            "export_path": "results/optimization/"
        }
    }
    
    with open(samples_dir / "optimization_example.yaml", "w") as f:
        yaml.dump(optimization_config, f, default_flow_style=False)
    
    # Sample live trading config
    live_config = {
        "name": "Momentum Strategy Live",
        "type": "live_trading",
        "description": "Live trading with momentum strategy",
        "paper_trading": True,
        "broker": {
            "name": "paper",
            "timeout": 30,
            "retry_count": 3
        },
        "data": {
            "provider": "yahoo",
            "symbols": ["SPY", "QQQ", "IWM", "DIA"],
            "frequency": "5min",
            "lookback_days": 30,
            "warmup_period": 100
        },
        "portfolio": {
            "initial_capital": 100000,
            "max_positions": 2,
            "position_sizing": "risk_parity"
        },
        "strategies": [{
            "name": "momentum",
            "type": "momentum_strategy",
            "allocation": 1.0,
            "parameters": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": -0.01,
                "use_volume": True
            }
        }],
        "risk": {
            "check_frequency": "pre_trade",
            "position_sizers": [{
                "name": "volatility_sized",
                "type": "volatility",
                "risk_per_trade": 1.0,
                "lookback_period": 20
            }],
            "limits": [{
                "type": "position",
                "max_position": 25000
            }, {
                "type": "exposure",
                "max_exposure_pct": 80
            }, {
                "type": "daily_loss",
                "max_daily_loss_pct": 2.0
            }, {
                "type": "symbol_restriction",
                "blocked_symbols": ["UVXY", "VXX"]
            }],
            "stop_loss": {
                "enabled": True,
                "type": "atr",
                "value": 2.0
            },
            "emergency_shutdown": {
                "enabled": True,
                "max_daily_loss_pct": 5.0,
                "max_drawdown_pct": 15.0
            }
        },
        "execution": {
            "order_type": "limit",
            "time_in_force": "DAY",
            "retry_failed_orders": True,
            "max_retry": 3,
            "split_orders": {
                "enabled": True,
                "max_order_size": 10000
            }
        },
        "monitoring": {
            "heartbeat_interval": 60,
            "log_level": "INFO",
            "alerts": {
                "webhook": "https://hooks.slack.com/services/xxx"
            },
            "metrics_export": {
                "enabled": True,
                "interval": 300,
                "path": "metrics/live_trading.csv"
            }
        }
    }
    
    with open(samples_dir / "live_trading_example.yaml", "w") as f:
        yaml.dump(live_config, f, default_flow_style=False)
    
    print(f"Sample configurations created in {samples_dir}")


def test_invalid_configs():
    """Test validation with invalid configurations."""
    validator = ConfigSchemaValidator()
    validator.register_schema(BACKTEST_SCHEMA)
    
    # Invalid config - missing required fields
    invalid_config1 = {
        "name": "Invalid Backtest",
        "strategies": [{
            "name": "test",
            "type": "test_strategy"
        }]
        # Missing: data, portfolio
    }
    
    result1 = validator.validate(invalid_config1, "backtest")
    print("\nTest 1 - Missing required fields:")
    print(f"Valid: {result1.is_valid}")
    print(f"Errors: {result1.errors}")
    
    # Invalid config - wrong date order
    invalid_config2 = {
        "name": "Invalid Date Range",
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2023-12-31",
            "end_date": "2023-01-01",  # End before start
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 100000
        },
        "strategies": [{
            "name": "test",
            "type": "test_strategy",
            "allocation": 0.5  # Doesn't sum to 1.0
        }, {
            "name": "test2",
            "type": "test_strategy2",
            "allocation": 0.3  # Total = 0.8
        }]
    }
    
    result2 = validator.validate(invalid_config2, "backtest")
    print("\nTest 2 - Invalid dates and allocations:")
    print(f"Valid: {result2.is_valid}")
    print(f"Errors: {result2.errors}")
    print(f"Warnings: {result2.warnings}")


if __name__ == "__main__":
    # Run examples
    print("=== Configuration Validator Examples ===\n")
    
    print("1. Validating a configuration dictionary:")
    validate_config_dict()
    
    print("\n2. Creating sample configurations:")
    create_sample_configs()
    
    print("\n3. Testing invalid configurations:")
    test_invalid_configs()
    
    print("\n4. Validating a YAML file:")
    # Uncomment to test with a real file
    # validate_yaml_file("configs/optimization_workflow.yaml")