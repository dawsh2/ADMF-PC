"""
Storage module for ADMF-PC.

Provides sparse storage for signals, classifier states, and results.
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