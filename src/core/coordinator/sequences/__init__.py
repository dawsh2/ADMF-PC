"""
Sequence patterns for phase execution.

Sequences define HOW phases are executed:
- single_pass: Execute once
- walk_forward: Rolling window execution
- monte_carlo: Multiple iterations with randomization
- parallel: Execute phases in parallel
"""

from .single_pass import SinglePassSequence
from .walk_forward import WalkForwardSequence
from .monte_carlo import MonteCarloSequence
from .train_test import TrainTestSequence

# Registry of available sequences
SEQUENCE_REGISTRY = {
    'single_pass': SinglePassSequence,
    'walk_forward': WalkForwardSequence,
    'monte_carlo': MonteCarloSequence,
    'train_test': TrainTestSequence
}

__all__ = [
    'SEQUENCE_REGISTRY',
    'SinglePassSequence',
    'WalkForwardSequence',
    'MonteCarloSequence',
    'TrainTestSequence'
]
