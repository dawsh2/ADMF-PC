"""
All tracing-related functionality - 100% feature parity.
"""

# Core tracing (from observers/ and storage/)
from .observers import EventTracer, MetricsObserver, create_tracer_from_config
from .storage import MemoryEventStorage, DiskEventStorage, create_storage_backend

# Hierarchical storage for workspace structure
from ..storage.hierarchical import HierarchicalEventStorage, HierarchicalStorageConfig
from .filters import (
    combine_filters, any_of_filters, strategy_filter, symbol_filter,
    classification_filter, strength_filter, metadata_filter, payload_filter,
    custom_filter, create_portfolio_filter
)

# Result extraction (ESSENTIAL!) - from extraction.py
from .extraction import MetricsExtractor, ObjectiveFunctionExtractor

# Event analysis (ESSENTIAL!) - from query.py
from .query import EventQueryInterface

# Flow validation (ESSENTIAL!) - from validation.py
from .validation import EventFlowValidator

# Signal storage & replay (ESSENTIAL!) - from storage/signals.py
from .signals import (
    SignalIndex, ClassifierChangeIndex, MultiSymbolSignal, 
    SignalStorageManager, SignalStorageProtocol, ClassifierStateProtocol
)

__all__ = [
    # Core tracing
    'EventTracer', 'MetricsObserver', 'create_tracer_from_config',
    'MemoryEventStorage', 'DiskEventStorage', 'create_storage_backend',
    'HierarchicalEventStorage', 'HierarchicalStorageConfig',
    'combine_filters', 'any_of_filters', 'strategy_filter', 'symbol_filter',
    'classification_filter', 'strength_filter', 'metadata_filter', 'payload_filter',
    'custom_filter', 'create_portfolio_filter',
    
    # Analysis features
    'MetricsExtractor', 'ObjectiveFunctionExtractor',
    'EventQueryInterface', 'EventFlowValidator',
    
    # Signal storage & replay
    'SignalIndex', 'ClassifierChangeIndex', 'MultiSymbolSignal', 
    'SignalStorageManager', 'SignalStorageProtocol', 'ClassifierStateProtocol'
]