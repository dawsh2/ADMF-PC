"""
Coordinator package with declarative pattern-based architecture.

This package contains the coordinator system that:
- Loads patterns from YAML files
- Supports data-driven workflows
- Enables zero-code configuration
- Follows protocol-based design
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