"""Execution module for order processing and trade execution.

Architecture Reference: docs/SYSTEM_ARCHITECTURE_V5.MD#execution-module
Style Guide: STYLE.md - Canonical execution implementations

This module provides Protocol + Composition execution engine with zero inheritance."""

from .protocols import (
    Broker,
    OrderProcessor,
    ExecutionEngine,
    MarketSimulator,
    OrderStatus,
    FillStatus,
)
from .brokers import SimulatedBroker, create_simulated_broker
from .order_manager import OrderManager
from .engine import DefaultExecutionEngine, create_execution_engine
from .brokers import (
    SlippageModel,
    CommissionModel,
    PercentageSlippageModel,
    TieredCommissionModel
)
# from .context import ExecutionContext  # Removed - moved to tmp/execution_cleanup/
# from .capabilities import ExecutionCapabilities  # Removed - capabilities.py doesn't exist
# Note: modes moved to core/coordinator/workflows/modes/
# Note: analysis moved to tmp/analysis/signal_analysis
# Note: containers_pipeline moved to core/coordinator/workflows/

__all__ = [
    # Protocols
    "Broker",
    "OrderProcessor",
    "ExecutionEngine",
    "MarketSimulator",
    "OrderStatus",
    "FillStatus",
    # Implementations
    "SimulatedBroker",
    "create_simulated_broker",
    "OrderManager",
    "DefaultExecutionEngine",
    "SlippageModel",
    "CommissionModel",
    "PercentageSlippageModel",
    "TieredCommissionModel",
    # "ExecutionContext",  # Removed - moved to tmp/execution_cleanup/
    # "ExecutionCapabilities",  # Removed - capabilities.py doesn't exist
    # Moved to coordinator
    # "BacktestConfig",
    # "BacktestResults",
]