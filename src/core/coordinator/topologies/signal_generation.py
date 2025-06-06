"""
Signal generation topology: Generate and save signals.

Pipeline: data → features → strategies (save signals to disk)
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from ...containers.container import Container, ContainerConfig, ContainerRole
from ...containers.components import DataStreamer, FeatureCalculator
from ...events.tracing import TracedEventBus, EventTracer
from ...events import EventBus

logger = logging.getLogger(__name__)


def create_signal_generation_topology(config: Dict[str, Any], tracing_enabled: bool = True) -> Dict[str, Any]:
    """
    Create topology for signal generation only.
    
    Creates:
    - Symbol-Timeframe containers for data and features
    - Stateless strategy components
    - NO Portfolio containers (signals saved to disk)
    - NO Execution container
    
    Returns:
        Dictionary with containers, components, and metadata
    """
    # Create root event bus
    if tracing_enabled:
        root_event_bus = TracedEventBus("root_event_bus")
        correlation_id = config.get('correlation_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        event_tracer = EventTracer(
            correlation_id=correlation_id,
            max_events=config.get('tracing', {}).get('max_events', 10000)
        )
        root_event_bus.set_tracer(event_tracer)
        logger.info("🔍 Created TracedEventBus for signal generation")
    else:
        root_event_bus = EventBus("root_event_bus")
        event_tracer = None
        
    topology = {
        'containers': {},
        'stateless_components': {},
        'parameter_combinations': [],
        'root_event_bus': root_event_bus,
        'event_tracer': event_tracer
    }
    
    # Extract symbol-timeframe configurations
    symbol_timeframe_configs = _extract_symbol_timeframe_configs(config)
    
    # Create Symbol-Timeframe containers (same as backtest)
    for st_config in symbol_timeframe_configs:
        symbol = st_config['symbol']
        timeframe = st_config.get('timeframe', '1d')
        
        # Data container
        data_container_id = f"{symbol}_{timeframe}_data"
        data_container = Container(ContainerConfig(
            role=ContainerRole.DATA,
            name=data_container_id,
            container_id=data_container_id
        ))
        
        data_container.add_component('data_streamer', DataStreamer(
            symbol=symbol,
            timeframe=timeframe,
            data_source=st_config.get('data_config', {})
        ))
        
        if tracing_enabled and event_tracer:
            _replace_event_bus_with_traced(data_container, data_container_id, event_tracer)
            
        topology['containers'][data_container_id] = data_container
        
        # Feature container
        feature_container_id = f"{symbol}_{timeframe}_features"
        feature_container = Container(ContainerConfig(
            role=ContainerRole.FEATURE,
            name=feature_container_id,
            container_id=feature_container_id
        ))
        
        feature_config = st_config.get('features', {})
        feature_container.add_component('feature_calculator', FeatureCalculator(
            indicators=feature_config.get('indicators', []),
            lookback_window=feature_config.get('lookback_window', 100)
        ))
        
        if tracing_enabled and event_tracer:
            _replace_event_bus_with_traced(feature_container, feature_container_id, event_tracer)
            
        topology['containers'][feature_container_id] = feature_container
    
    # Wire Data → Feature flow
    for st_config in symbol_timeframe_configs:
        symbol = st_config['symbol']
        timeframe = st_config.get('timeframe', '1d')
        data_container_id = f"{symbol}_{timeframe}_data"
        feature_container_id = f"{symbol}_{timeframe}_features"
        
        data_container = topology['containers'][data_container_id]
        feature_container = topology['containers'][feature_container_id]
        
        feature_calc = feature_container.get_component('feature_calculator')
        if feature_calc and hasattr(feature_calc, 'on_bar'):
            data_container.event_bus.subscribe('BAR', feature_calc.on_bar)
            logger.info(f"Wired {data_container_id} → {feature_container_id}")
    
    # Create stateless components
    from .helpers.component_builder import create_stateless_components
    topology['stateless_components'] = create_stateless_components(config)
    
    # Expand parameter combinations (for signal generation variations)
    from .backtest import _expand_parameter_combinations
    param_combos = _expand_parameter_combinations(config)
    topology['parameter_combinations'] = param_combos
    
    # NO Portfolio or Execution containers for signal generation
    
    logger.info(f"Signal generation topology created: {len(topology['containers'])} containers")
    
    # Wire containers
    from .helpers.adapter_wiring import wire_signal_generation_topology
    
    wiring_config = {
        'root_event_bus': root_event_bus,
        'stateless_components': topology['stateless_components'],
        'parameter_combinations': topology['parameter_combinations']
    }
    
    adapters = wire_signal_generation_topology(topology['containers'], wiring_config)
    topology['adapters'] = adapters
    
    return topology


# Import helper functions from backtest topology
from .backtest import (
    _extract_symbol_timeframe_configs,
    _replace_event_bus_with_traced
)