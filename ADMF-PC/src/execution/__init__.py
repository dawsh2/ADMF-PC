"""Execution module for order processing and trade execution."""

from .protocols import (
    Broker,
    OrderProcessor,
    ExecutionEngine,
    MarketSimulator,
    ExecutionCapability,
)
from .backtest_broker import BacktestBroker
from .backtest_broker_refactored import BacktestBrokerRefactored
from .order_manager import OrderManager
from .execution_engine import DefaultExecutionEngine
from .market_simulation import MarketSimulator, SlippageModel, CommissionModel
from .execution_context import ExecutionContext
from .capabilities import ExecutionCapabilities
from .backtest_engine import UnifiedBacktestEngine, BacktestConfig, BacktestResults

__all__ = [
    # Protocols
    "Broker",
    "OrderProcessor",
    "ExecutionEngine",
    "MarketSimulator",
    "ExecutionCapability",
    # Implementations
    "BacktestBroker",
    "BacktestBrokerRefactored",
    "OrderManager",
    "DefaultExecutionEngine",
    "MarketSimulator",
    "SlippageModel",
    "CommissionModel",
    "ExecutionContext",
    "ExecutionCapabilities",
    # Backtesting
    "UnifiedBacktestEngine",
    "BacktestConfig",
    "BacktestResults",
]