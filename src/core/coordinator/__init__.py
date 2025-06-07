"""
Refactored Coordinator with Protocol + Composition Architecture

This package contains the refactored coordinator system that follows:
- Protocol-based design (no inheritance)
- Composition over inheritance
- Clean separation of concerns
- Decorator-based discovery
"""

from .coordinator import Coordinator
from .sequencer import Sequencer
from .topology import TopologyBuilder
from .protocols import WorkflowProtocol, SequenceProtocol, PhaseConfig

__all__ = [
    'Coordinator',
    'Sequencer', 
    'TopologyBuilder',
    'WorkflowProtocol',
    'SequenceProtocol',
    'PhaseConfig'
]
