"""Test the configuration schema validator."""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import (
    ConfigSchemaValidator,
    BACKTEST_SCHEMA,
    OPTIMIZATION_SCHEMA,
    LIVE_TRADING_SCHEMA
)


def test_backtest_validation():
    """Test backtest configuration validation."""
    print("\n=== Testing Backtest Configuration Validation ===")
    
    validator = ConfigSchemaValidator()
    validator.register_schema(BACKTEST_SCHEMA)
    
    # Valid configuration
    valid_config = {
        "name": "Test Backtest",
        "type": "backtest",
        "data": {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 100000,
            "commission": {
                "type": "fixed",
                "value": 1.0
            }
        },
        "strategies": [{
            "name": "test_strategy",
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
            }]
        }
    }
    
    result = validator.validate(valid_config, "backtest")
    print(f"\nValid config test:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    
    # Invalid configuration - missing required field
    invalid_config = {
        "name": "Invalid Backtest",
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d"
        }
        # Missing: portfolio, strategies
    }
    
    result = validator.validate(invalid_config, "backtest")
    print(f"\nInvalid config test (missing fields):")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test with invalid date range
    date_test_config = {
        "name": "Date Test",
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2023-12-31",
            "end_date": "2023-01-01",  # End before start
            "frequency": "1d"
        },
        "portfolio": {"initial_capital": 100000},
        "strategies": [{"name": "test", "type": "test"}]
    }
    
    result = validator.validate(date_test_config, "backtest")
    print(f"\nInvalid date range test:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")


def test_optimization_validation():
    """Test optimization configuration validation."""
    print("\n=== Testing Optimization Configuration Validation ===")
    
    validator = ConfigSchemaValidator()
    validator.register_schema(OPTIMIZATION_SCHEMA)
    
    # Valid configuration
    valid_config = {
        "name": "Test Optimization",
        "type": "optimization",
        "base_config": {
            "data": {
                "symbols": ["SPY"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "frequency": "1d"
            },
            "portfolio": {"initial_capital": 100000},
            "strategies": [{"name": "test", "type": "test_strategy"}]
        },
        "optimization": {
            "method": "bayesian",
            "parameter_space": {
                "param1": {"type": "int", "min": 1, "max": 100},
                "param2": {"type": "float", "min": 0.1, "max": 1.0}
            },
            "objectives": [{
                "metric": "sharpe_ratio",
                "direction": "maximize"
            }],
            "n_trials": 50
        }
    }
    
    result = validator.validate(valid_config, "optimization")
    print(f"\nValid optimization config:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test with invalid parameter constraints
    param_test_config = {
        "name": "Param Test",
        "base_config": {
            "data": {"symbols": ["SPY"], "start_date": "2023-01-01", 
                    "end_date": "2023-12-31", "frequency": "1d"},
            "portfolio": {"initial_capital": 100000},
            "strategies": [{"name": "test", "type": "test"}]
        },
        "optimization": {
            "method": "grid_search",
            "parameter_space": {
                "fast_period": {"type": "int", "min": 50, "max": 10},  # min > max
                "slow_period": {"type": "int", "min": 20, "max": 200}
            },
            "objectives": [{"metric": "returns", "direction": "maximize"}]
        }
    }
    
    result = validator.validate(param_test_config, "optimization")
    print(f"\nInvalid parameter constraints test:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")


def test_live_trading_validation():
    """Test live trading configuration validation."""
    print("\n=== Testing Live Trading Configuration Validation ===")
    
    validator = ConfigSchemaValidator()
    validator.register_schema(LIVE_TRADING_SCHEMA)
    
    # Valid paper trading config
    valid_config = {
        "name": "Test Live Trading",
        "type": "live_trading",
        "paper_trading": True,
        "broker": {
            "name": "paper"
        },
        "data": {
            "provider": "yahoo",
            "symbols": ["SPY", "QQQ"],
            "frequency": "5min"
        },
        "portfolio": {
            "initial_capital": 100000
        },
        "strategies": [{
            "name": "test_strategy",
            "type": "momentum"
        }]
    }
    
    result = validator.validate(valid_config, "live_trading")
    print(f"\nValid paper trading config:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test live trading without API key
    live_config = {
        "name": "Live Test",
        "type": "live_trading",
        "paper_trading": False,  # Real trading
        "broker": {
            "name": "alpaca"
            # Missing: api_key, api_secret
        },
        "data": {
            "provider": "alpaca",
            "symbols": ["SPY"],
            "frequency": "1min"
        },
        "portfolio": {"initial_capital": 100000},
        "strategies": [{"name": "test", "type": "test"}]
    }
    
    result = validator.validate(live_config, "live_trading")
    print(f"\nLive trading without API key test:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test with risk warnings
    risk_config = {
        "name": "Risk Test",
        "paper_trading": True,
        "broker": {"name": "paper"},
        "data": {
            "provider": "yahoo",
            "symbols": ["SPY"],
            "frequency": "1min"
        },
        "portfolio": {"initial_capital": 100000},
        "strategies": [{"name": "test", "type": "test"}],
        "risk": {
            "limits": [{
                "type": "position",
                "max_position": 50000
            }]
            # No daily loss limit - should generate warning
        }
    }
    
    result = validator.validate(risk_config, "live_trading")
    print(f"\nRisk configuration test:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Warnings: {result.warnings}")


def test_schema_detection():
    """Test automatic schema detection."""
    print("\n=== Testing Schema Detection ===")
    
    validator = ConfigSchemaValidator()
    validator.register_schema(BACKTEST_SCHEMA)
    validator.register_schema(OPTIMIZATION_SCHEMA)
    validator.register_schema(LIVE_TRADING_SCHEMA)
    
    # Test configs without explicit type
    configs_to_test = [
        {
            "name": "Backtest Detection",
            "backtesting": {"enabled": True},  # Should detect as backtest
            "data": {"symbols": ["AAPL"], "start_date": "2023-01-01", 
                    "end_date": "2023-12-31", "frequency": "1d"},
            "portfolio": {"initial_capital": 100000},
            "strategies": [{"name": "test", "type": "test"}]
        },
        {
            "name": "Optimization Detection",
            "optimization": {"method": "bayesian"},  # Should detect as optimization
            "base_config": {
                "data": {"symbols": ["SPY"], "start_date": "2023-01-01",
                        "end_date": "2023-12-31", "frequency": "1d"},
                "portfolio": {"initial_capital": 100000},
                "strategies": [{"name": "test", "type": "test"}]
            },
            "optimization": {
                "method": "bayesian",
                "parameter_space": {"p1": {"type": "int", "min": 1, "max": 10}},
                "objectives": [{"metric": "sharpe", "direction": "maximize"}]
            }
        },
        {
            "name": "Live Trading Detection",
            "broker": {"name": "paper"},  # Should detect as live trading
            "data": {"provider": "yahoo", "symbols": ["SPY"], "frequency": "1min"},
            "portfolio": {"initial_capital": 100000},
            "strategies": [{"name": "test", "type": "test"}]
        }
    ]
    
    for config in configs_to_test:
        # Use internal method for testing
        detected = validator._detect_schema(config)
        print(f"\nConfig '{config['name']}':")
        print(f"  Detected schema: {detected}")


def test_yaml_file_validation():
    """Test validating actual YAML files."""
    print("\n=== Testing YAML File Validation ===")
    
    # Create a test YAML file
    test_dir = Path("test_configs")
    test_dir.mkdir(exist_ok=True)
    
    test_config = {
        "name": "Test MA Crossover",
        "type": "backtest",
        "data": {
            "symbols": ["SPY", "QQQ"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 100000
        },
        "strategies": [{
            "name": "ma_cross",
            "type": "moving_average_crossover",
            "parameters": {
                "fast_period": 10,
                "slow_period": 30
            }
        }]
    }
    
    test_file = test_dir / "test_backtest.yaml"
    with open(test_file, "w") as f:
        yaml.dump(test_config, f)
    
    # Validate the file
    validator = ConfigSchemaValidator()
    validator.register_schema(BACKTEST_SCHEMA)
    
    result = validator.validate_file(test_file)
    print(f"\nYAML file validation:")
    print(f"  File: {test_file}")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test with non-existent file
    result = validator.validate_file("non_existent.yaml")
    print(f"\nNon-existent file test:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Clean up
    test_file.unlink()
    test_dir.rmdir()


def test_normalization():
    """Test configuration normalization."""
    print("\n=== Testing Configuration Normalization ===")
    
    validator = ConfigSchemaValidator()
    validator.register_schema(BACKTEST_SCHEMA)
    
    # Minimal config that should get defaults
    minimal_config = {
        "name": "Minimal Test",
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 50000
        },
        "strategies": [{
            "name": "test",
            "type": "test_strategy"
        }]
    }
    
    result = validator.validate(minimal_config, "backtest")
    print(f"\nNormalized configuration:")
    print(f"  Is valid: {result.is_valid}")
    
    if result.normalized_config:
        print(f"  Added defaults:")
        print(f"    - execution_mode: {result.normalized_config.get('execution_mode')}")
        print(f"    - data.format: {result.normalized_config['data'].get('format')}")
        print(f"    - data.timezone: {result.normalized_config['data'].get('timezone')}")
        print(f"    - strategies[0].enabled: {result.normalized_config['strategies'][0].get('enabled')}")
        print(f"    - strategies[0].allocation: {result.normalized_config['strategies'][0].get('allocation')}")


if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Schema Validator Tests")
    print("=" * 60)
    
    test_backtest_validation()
    test_optimization_validation()
    test_live_trading_validation()
    test_schema_detection()
    test_yaml_file_validation()
    test_normalization()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)