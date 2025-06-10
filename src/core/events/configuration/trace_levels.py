"""Sophisticated trace configuration with presets from old implementation."""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import fnmatch

class TraceLevel(Enum):
    """Enhanced trace level presets."""
    NONE = "none"               # Zero overhead for production
    MINIMAL = "minimal"         # Minimal tracing for optimization
    METRICS = "metrics"         # Only metrics, no event storage
    TRADES = "trades"          # Metrics + completed trades
    EQUITY_CURVE = "equity"    # Metrics + trades + equity snapshots
    NORMAL = "normal"          # Standard tracing for development
    DEBUG = "debug"            # Full debug tracing
    FULL = "full"              # Complete event trace
    CUSTOM = "custom"          # User-defined settings

@dataclass
class TraceLevelConfig:
    """Configuration for a trace level."""
    enabled: bool
    max_events: int
    retention_policy: str
    trace_pattern: Optional[str] = None
    description: str = ""
    container_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class TraceConfiguration:
    """
    Container-specific trace configuration with wildcard support.
    
    Critical for memory management in large-scale optimizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.default_level = TraceLevel(config.get('default', 'metrics'))
        self.container_overrides = config.get('overrides', {})
        
    def get_trace_level(self, container_id: str) -> TraceLevel:
        """Get trace level for specific container using wildcards."""
        # Check exact match first
        if container_id in self.container_overrides:
            return TraceLevel(self.container_overrides[container_id])
            
        # Check wildcard patterns
        for pattern, level in self.container_overrides.items():
            if fnmatch.fnmatch(container_id, pattern):
                return TraceLevel(level)
                
        return self.default_level
    
    def should_trace_event(self, container_id: str, event_type: str) -> bool:
        """Check if specific event type should be traced for container."""
        trace_level = self.get_trace_level(container_id)
        
        if trace_level == TraceLevel.NONE:
            return False
            
        if trace_level == TraceLevel.FULL:
            return True
            
        # Define which events are traced at each level
        if trace_level == TraceLevel.METRICS:
            return event_type in ['PORTFOLIO_UPDATE', 'POSITION_CLOSE']
            
        if trace_level == TraceLevel.TRADES:
            return event_type in ['PORTFOLIO_UPDATE', 'POSITION_OPEN', 
                                 'POSITION_CLOSE', 'FILL']
            
        if trace_level == TraceLevel.EQUITY_CURVE:
            return event_type in ['PORTFOLIO_UPDATE', 'POSITION_OPEN', 
                                 'POSITION_CLOSE', 'FILL', 'ORDER']
            
        return False

# Sophisticated trace level presets from old implementation
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
                "store_trades": False,  # Don't store trade history
                "store_equity_curve": False  # Don't store equity curve
            },
            # ALL other containers disabled for minimal
            "data_*": {"enabled": False},
            "feature_*": {"enabled": False},
            "strategy_*": {"enabled": False},
            "risk_*": {"enabled": False},
            "execution_*": {"enabled": False}
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
                "store_trades": True,  # Store trade history
                "store_equity_curve": True,  # Store equity snapshots
                "snapshot_interval": 100
            },
            # Data containers: limited tracing
            "data_*": {"enabled": True, "max_events": 1000},
            # Feature containers: moderate tracing
            "feature_*": {"enabled": True, "max_events": 5000},
            # Strategy containers: still stateless, minimal
            "strategy_*": {"enabled": True, "max_events": 1000},
            # Execution container: full order tracking
            "execution_*": {"enabled": True, "max_events": 5000}
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
                "store_trades": True,
                "store_equity_curve": True,
                "snapshot_interval": 10  # Frequent snapshots
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
    for pattern, settings in preset.container_overrides.items():
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
    
    return config

def apply_trace_level(config: Dict[str, Any], trace_level: str) -> Dict[str, Any]:
    """
    Apply a trace level preset to existing configuration.
    
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

# Example configuration for optimization
DEFAULT_TRACE_CONFIG = {
    'default': 'minimal',  # Most containers only need minimal tracing
    'overrides': {
        'portfolio_*': 'metrics',      # Portfolios track metrics only
        'best_portfolio_*': 'trades',  # Best performers get trade details
        'analysis_*': 'full',          # Analysis containers get everything
        'data_*': 'none'               # Data containers need no tracing
    }
}