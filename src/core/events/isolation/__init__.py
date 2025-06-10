"""
Event isolation for parallel execution and thread safety.
"""

from .thread_local import EventIsolationManager

__all__ = [
    'EventIsolationManager'
]