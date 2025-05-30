"""Execution module for order processing and trade execution."""

from .protocols import (
    Broker,
    OrderProcessor,
    ExecutionEngine,
    MarketSimulator,
    ExecutionCapability,
    OrderStatus,
    FillStatus,
)
from .backtest_broker import BacktestBroker
from .backtest_broker_refactored import BacktestBrokerRefactored
from .order_manager import OrderManager
from .execution_engine import DefaultExecutionEngine
from .market_simulation import (
    MarketSimulator,
    SlippageModel,
    CommissionModel,
    PercentageSlippageModel,
    TieredCommissionModel
)
from .execution_context import ExecutionContext
from .capabilities import ExecutionCapabilities
from .backtest_engine import UnifiedBacktestEngine, BacktestConfig, BacktestResults
from .signal_generation_engine import SignalGenerationContainer, SignalGenerationContainerFactory
from .signal_replay_engine import SignalReplayContainer, SignalReplayContainerFactory
from .analysis import SignalAnalysisEngine, SignalAnalysisResult, AnalysisType

__all__ = [
    # Protocols
    "Broker",
    "OrderProcessor",
    "ExecutionEngine",
    "MarketSimulator",
    "ExecutionCapability",
    "OrderStatus",
    "FillStatus",
    # Implementations
    "BacktestBroker",
    "BacktestBrokerRefactored",
    "OrderManager",
    "DefaultExecutionEngine",
    "MarketSimulator",
    "SlippageModel",
    "CommissionModel",
    "PercentageSlippageModel",
    "TieredCommissionModel",
    "ExecutionContext",
    "ExecutionCapabilities",
    # Backtesting
    "UnifiedBacktestEngine",
    "BacktestConfig",
    "BacktestResults",
    # Signal Analysis
    "SignalGenerationContainer",
    "SignalGenerationContainerFactory",
    "SignalReplayContainer",
    "SignalReplayContainerFactory",
    "SignalAnalysisEngine",
    "SignalAnalysisResult",
    "AnalysisType",
]