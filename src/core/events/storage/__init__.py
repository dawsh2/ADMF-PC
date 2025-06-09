"""
Event storage components for signal replay.

This module provides sparse storage for signals and classifier states,
enabling efficient replay of trading strategies without recomputation.
"""

from .signals import (
    SignalStorageManager,
    SignalIndex,
    ClassifierChangeIndex,
    MultiSymbolSignal,
    SignalStorageProtocol,
    ClassifierStateProtocol
)

__all__ = [
    'SignalStorageManager',
    'SignalIndex', 
    'ClassifierChangeIndex',
    'MultiSymbolSignal',
    'SignalStorageProtocol',
    'ClassifierStateProtocol'
]
