"""
Backtest module for ADMF-PC.

Separates backtest execution concerns from workflow orchestration.
The Coordinator handles high-level workflow management while the
BacktestEngine handles the details of running strategies against data.
"""

from .backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    Position,
    Trade
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'Position',
    'Trade'
]