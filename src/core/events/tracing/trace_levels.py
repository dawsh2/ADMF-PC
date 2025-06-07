"""
Trace Level Presets for Event Tracing

Provides pre-configured trace levels for common scenarios to simplify
event tracing configuration.
"""

from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class TraceLevel(Enum):
    """Standard trace level presets."""
    NONE = "none"          # No tracing - zero overhead
    MINIMAL = "minimal"    # Minimal tracing for optimization
    NORMAL = "normal"      # Standard tracing 
    DEBUG = "debug"        # Full debug tracing
    CUSTOM = "custom"      # User-defined settings


@dataclass
class TraceLevelConfig:
    """Configuration for a trace level."""
    enabled: bool
    max_events: int
    retention_policy: str
    trace_pattern: Optional[str] = None
    description: str = ""
    container_overrides: Dict[str, Dict[str, Any]] = None


# Trace level presets
TRACE_LEVEL_PRESETS: Dict[TraceLevel, TraceLevelConfig] = {
    TraceLevel.NONE: TraceLevelConfig(
        enabled=False,
        max_events=0,
        retention_policy="none",
        description="No tracing - zero overhead for production"
    ),
    
    TraceLevel.MINIMAL: TraceLevelConfig(
        enabled=True,
        max_events=0,  # No global limit
        retention_policy="trade_complete",  # Only keep events for open trades
        trace_pattern=None,  # Need all events to track trades properly
        description="Minimal tracing - only tracks open trades then discards",
        container_overrides={
            # Portfolio containers: track trades until complete, then discard
            "portfolio_*": {
                "enabled": True,
                "max_events": 0,  # No limit - let retention policy handle cleanup
                "retention_policy": "trade_complete",
                "results": {
                    "streaming_metrics": True,  # Keep metrics in memory
                    "retention_policy": "trade_complete",
                    "store_trades": False,  # Don't store trade history
                    "store_equity_curve": False  # Don't store equity curve
                }
            },
            # ALL other containers disabled for minimal
            "data_*": {
                "enabled": False,
            },
            "feature_*": {
                "enabled": False,
            },
            "strategy_*": {
                "enabled": False,
            },
            "risk_*": {
                "enabled": False,
            },
            "execution_*": {
                "enabled": False,  # Not needed for minimal
            }
        }
    ),
    
    TraceLevel.NORMAL: TraceLevelConfig(
        enabled=True,
        max_events=10000,
        retention_policy="trade_complete",
        trace_pattern=None,  # Trace all events
        description="Standard tracing for development",
        container_overrides={
            # Portfolio containers: full metrics and trade history
            "portfolio_*": {
                "enabled": True,
                "max_events": 10000,
                "retention_policy": "trade_complete",
                "results": {
                    "streaming_metrics": True,
                    "retention_policy": "trade_complete",
                    "store_trades": True,  # Store trade history
                    "store_equity_curve": True,  # Store equity snapshots
                    "snapshot_interval": 100
                }
            },
            # Data containers: limited tracing
            "data_*": {
                "enabled": True,
                "max_events": 1000,
            },
            # Feature containers: moderate tracing
            "feature_*": {
                "enabled": True,
                "max_events": 5000,
            },
            # Strategy containers: still stateless, minimal
            "strategy_*": {
                "enabled": True,
                "max_events": 1000,
            },
            # Execution container: full order tracking
            "execution_*": {
                "enabled": True,
                "max_events": 5000,
            }
        }
    ),
    
    TraceLevel.DEBUG: TraceLevelConfig(
        enabled=True,
        max_events=50000,
        retention_policy="sliding_window",
        trace_pattern=None,  # Trace everything
        description="Full tracing for debugging",
        container_overrides={
            # Portfolio containers: everything stored
            "portfolio_*": {
                "enabled": True,
                "max_events": 50000,
                "retention_policy": "sliding_window",
                "results": {
                    "streaming_metrics": True,
                    "retention_policy": "sliding_window",
                    "store_trades": True,
                    "store_equity_curve": True,
                    "snapshot_interval": 10  # Frequent snapshots
                }
            },
            # All containers get full tracing
            "*": {
                "enabled": True,
                "max_events": 50000,
                "retention_policy": "sliding_window"
            }
        }
    )
}


def get_trace_config(trace_level: str) -> Dict[str, Any]:
    """
    Get trace configuration for a given trace level.
    
    Args:
        trace_level: Trace level name (none, minimal, normal, debug)
        
    Returns:
        Dictionary with trace configuration settings
    """
    # Handle string input
    try:
        level = TraceLevel(trace_level.lower())
    except ValueError:
        # Default to minimal for unknown levels
        level = TraceLevel.MINIMAL
    
    preset = TRACE_LEVEL_PRESETS[level]
    
    # Process container overrides to extract results config
    container_settings = {}
    for pattern, settings in (preset.container_overrides or {}).items():
        container_config = settings.copy()
        # Extract results config if present
        results_config = container_config.pop('results', None)
        container_settings[pattern] = container_config
    
    config = {
        'enable_event_tracing': preset.enabled,
        'trace_settings': {
            'max_events': preset.max_events,
            'retention_policy': preset.retention_policy,
            'trace_pattern': preset.trace_pattern,
            'container_settings': container_settings
        }
    }
    
    # Add results configuration for portfolio containers if specified
    if preset.container_overrides:
        for pattern, settings in preset.container_overrides.items():
            if 'results' in settings and 'portfolio' in pattern:
                if 'results' not in config:
                    config['results'] = {}
                # Apply results settings based on trace level
                config['results'].update(settings['results'])
    
    return config


def apply_trace_level(config: Dict[str, Any], trace_level: str) -> Dict[str, Any]:
    """
    Apply a trace level preset to existing configuration.
    
    This merges the trace level settings with existing config,
    allowing trace_level to be specified at the top level.
    
    Args:
        config: Existing configuration dictionary
        trace_level: Trace level to apply
        
    Returns:
        Updated configuration with trace settings
    """
    # Get trace config for level
    trace_config = get_trace_config(trace_level)
    
    # Merge into execution config
    if 'execution' not in config:
        config['execution'] = {}
    
    # Update execution config with trace settings
    config['execution'].update(trace_config)
    
    return config


def get_trace_level_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Extract trace level from configuration if specified.
    
    Looks for trace_level in:
    1. Top level config
    2. execution.trace_level
    3. execution.trace_settings.level
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trace level name or None
    """
    # Check top level
    if 'trace_level' in config:
        return config['trace_level']
    
    # Check execution config
    execution_config = config.get('execution', {})
    if 'trace_level' in execution_config:
        return execution_config['trace_level']
    
    # Check trace settings
    trace_settings = execution_config.get('trace_settings', {})
    if 'level' in trace_settings:
        return trace_settings['level']
    
    return None


def validate_trace_config(config: Dict[str, Any]) -> bool:
    """
    Validate trace configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    execution_config = config.get('execution', {})
    
    # If tracing is disabled, always valid
    if not execution_config.get('enable_event_tracing', False):
        return True
    
    trace_settings = execution_config.get('trace_settings', {})
    
    # Check required fields
    if 'max_events' not in trace_settings:
        return False
    
    # Validate max_events is reasonable
    max_events = trace_settings['max_events']
    if not isinstance(max_events, int) or max_events < 0:
        return False
    
    # Warn if max_events is very large
    if max_events > 100000:
        import logging
        logging.getLogger(__name__).warning(
            f"Large max_events setting ({max_events}) may consume significant memory"
        )
    
    return True