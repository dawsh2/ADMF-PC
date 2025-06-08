"""
Component registry for container factory.

Maps component type names to their implementations.
"""

from typing import Dict, Type, Any
from dataclasses import dataclass

# Import all available components
from .components import (
    DataStreamer,
    FeatureCalculator,
    PortfolioState,
    SignalProcessor,
    OrderGenerator,
    RiskValidator,
    ExecutionEngine
)

# Import components from separate files
from .components.signal_generator import SignalGeneratorComponent
from .components.signal_streamer import SignalStreamerComponent

# Import TimeAlignmentBuffer if available
try:
    from .multi_asset_timeframe_sync import TimeAlignmentBuffer
except ImportError:
    TimeAlignmentBuffer = None
    import logging
    logging.getLogger(__name__).debug("TimeAlignmentBuffer not available")


# Component registry mapping type names to classes
COMPONENT_REGISTRY: Dict[str, Type[Any]] = {
    # Data components
    'DataStreamer': DataStreamer,
    'CSVDataLoader': DataStreamer,  # Alias for backward compatibility
    
    # Feature components
    'FeatureCalculator': FeatureCalculator,
    'SignalGeneratorComponent': SignalGeneratorComponent,
    'SignalGenerator': SignalGeneratorComponent,  # Alias
    
    # Portfolio components
    'PortfolioState': PortfolioState,
    'SignalProcessor': SignalProcessor,
    'OrderGenerator': OrderGenerator,
    
    # Risk components
    'RiskValidator': RiskValidator,
    
    # Execution components
    'ExecutionEngine': ExecutionEngine,
    
    # Signal replay components
    'SignalStreamerComponent': SignalStreamerComponent,
    'SignalStreamer': SignalStreamerComponent,  # Alias
}

# Add TimeAlignmentBuffer if available
if TimeAlignmentBuffer is not None:
    COMPONENT_REGISTRY['TimeAlignmentBuffer'] = TimeAlignmentBuffer


def get_component_class(component_type: str) -> Type[Any]:
    """Get component class by type name."""
    if component_type not in COMPONENT_REGISTRY:
        raise ValueError(f"Unknown component type: {component_type}")
    return COMPONENT_REGISTRY[component_type]


def create_component(component_type: str, config: Dict[str, Any]) -> Any:
    """Create a component instance from configuration."""
    component_class = get_component_class(component_type)
    
    # Handle different initialization patterns
    if hasattr(component_class, 'from_config'):
        return component_class.from_config(config)
    else:
        # Use dataclass-style initialization
        return component_class(**config)