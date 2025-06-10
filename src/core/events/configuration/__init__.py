"""
Event system configuration including trace levels and presets.
"""

from .trace_levels import (
    TraceLevel, 
    TraceConfiguration, 
    TraceLevelConfig,
    TRACE_LEVEL_PRESETS,
    get_trace_config, 
    apply_trace_level,
    DEFAULT_TRACE_CONFIG
)

__all__ = [
    'TraceLevel', 
    'TraceConfiguration',
    'TraceLevelConfig',
    'TRACE_LEVEL_PRESETS',
    'get_trace_config',
    'apply_trace_level',
    'DEFAULT_TRACE_CONFIG'
]