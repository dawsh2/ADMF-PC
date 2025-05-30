"""Configuration validation and schema management."""

from .simple_validator import (
    SimpleConfigValidator as ConfigSchemaValidator,
    SchemaValidationError,
    ValidationResult,
)

# Export schemas as named constants for compatibility
BACKTEST_SCHEMA = "backtest"
OPTIMIZATION_SCHEMA = "optimization" 
LIVE_TRADING_SCHEMA = "live_trading"
STRATEGY_SCHEMA = "strategy"
RISK_SCHEMA = "risk"
DATA_SCHEMA = "data"
EXECUTION_SCHEMA = "execution"

__all__ = [
    'ConfigSchemaValidator',
    'SchemaValidationError',
    'ValidationResult',
    'BACKTEST_SCHEMA',
    'OPTIMIZATION_SCHEMA',
    'LIVE_TRADING_SCHEMA',
    'STRATEGY_SCHEMA',
    'RISK_SCHEMA',
    'DATA_SCHEMA',
    'EXECUTION_SCHEMA'
]