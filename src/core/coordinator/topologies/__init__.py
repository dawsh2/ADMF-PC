"""
Modular Topology Definitions for ADMF-PC

This package contains standardized topology definitions that can be easily modified
to support different pipeline arrangements and new components.

Each topology file defines:
- Pipeline structure (order of components)
- Container creation using generic Container + components pattern
- Adapter wiring for event flow
- Event flow configuration

Topologies:
- backtest: Full pipeline execution (data → features → strategies → portfolios → risk → execution)
- signal_generation: Signal generation only (data → features → strategies, save signals)
- signal_replay: Signal replay (saved signals → portfolios → risk → execution)
"""

from .backtest import create_backtest_topology
from .signal_generation import create_signal_generation_topology
from .signal_replay import create_signal_replay_topology

__all__ = [
    'create_backtest_topology',
    'create_signal_generation_topology',
    'create_signal_replay_topology'
]