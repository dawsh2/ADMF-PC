"""
Signal replay topology: Replay saved signals through execution.

Pipeline: saved signals → portfolios → risk → execution
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from ...containers.container import Container, ContainerConfig, ContainerRole
from ...containers.components import (
    SignalReplayComponent,
    PortfolioState,
    SignalProcessor,
    OrderGenerator,
    ExecutionEngine
)
from ...events.tracing import TracedEventBus, EventTracer
from ...events import EventBus

logger = logging.getLogger(__name__)


def create_signal_replay_topology(config: Dict[str, Any], tracing_enabled: bool = True) -> Dict[str, Any]:
    """
    Create topology for signal replay.
    
    Creates:
    - Signal replay container to load and emit saved signals
    - Portfolio containers for each parameter combination
    - Risk validation (stateless)
    - Execution container for order processing
    - NO Data containers
    - NO Feature containers
    - NO Strategy components
    
    Returns:
        Dictionary with containers, components, and metadata
    """
    # Create root event bus
    if tracing_enabled:
        root_event_bus = TracedEventBus("root_event_bus")
        correlation_id = config.get('correlation_id', f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        event_tracer = EventTracer(
            correlation_id=correlation_id,
            max_events=config.get('tracing', {}).get('max_events', 10000)
        )
        root_event_bus.set_tracer(event_tracer)
        logger.info("🔍 Created TracedEventBus for signal replay")
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
    
    # Create Signal Replay container
    signal_replay_config = config.get('signal_replay', {})
    replay_container_id = "signal_replay"
    replay_container = Container(ContainerConfig(
        role=ContainerRole.DATA,  # Using DATA role since it sources signals
        name=replay_container_id,
        container_id=replay_container_id
    ))
    
    # Add signal replay component
    replay_container.add_component('signal_replay', SignalReplayComponent(
        signal_file=signal_replay_config.get('signal_file', 'signals.pkl'),
        signal_directory=signal_replay_config.get('signal_directory', './results/signals/'),
        replay_speed=signal_replay_config.get('replay_speed', 1.0),
        filter_config=signal_replay_config.get('filter', {})
    ))
    
    if tracing_enabled and event_tracer:
        _replace_event_bus_with_traced(replay_container, replay_container_id, event_tracer)
        
    topology['containers'][replay_container_id] = replay_container
    
    # Expand parameter combinations (same as backtest)
    from .backtest import _expand_parameter_combinations
    param_combos = _expand_parameter_combinations(config)
    topology['parameter_combinations'] = param_combos
    
    # Create Portfolio containers (same structure as backtest)
    for combo in param_combos:
        combo_id = combo['combo_id']
        
        portfolio_container = Container(ContainerConfig(
            role=ContainerRole.PORTFOLIO,
            name=f'portfolio_{combo_id}',
            container_id=f'portfolio_{combo_id}'
        ))
        
        # Add portfolio components
        portfolio_config = config.get('portfolio', {})
        portfolio_container.add_component('portfolio_state', PortfolioState(
            initial_capital=portfolio_config.get('initial_capital', 100000),
            max_positions=portfolio_config.get('max_positions', 10),
            position_sizing_method=portfolio_config.get('position_sizing_method', 'equal_weight')
        ))
        
        portfolio_container.add_component('signal_processor', SignalProcessor())
        portfolio_container.add_component('order_generator', OrderGenerator())
        
        if tracing_enabled and event_tracer:
            _replace_event_bus_with_traced(portfolio_container, f"portfolio_{combo_id}", event_tracer)
            
        # Give portfolio access to root event bus
        portfolio_container.root_event_bus = root_event_bus
        
        topology['containers'][f'portfolio_{combo_id}'] = portfolio_container
    
    # Create Execution container (same as backtest)
    execution_container = Container(ContainerConfig(
        role=ContainerRole.EXECUTION,
        name='execution',
        container_id='execution'
    ))
    
    # Add execution engine component
    exec_config = config.get('execution', {})
    execution_container.add_component('execution_engine', ExecutionEngine(
        slippage_model=exec_config.get('slippage_model'),
        commission_model=exec_config.get('commission_model'),
        fill_latency=exec_config.get('fill_latency', 0),
        partial_fills=exec_config.get('partial_fills', False)
    ))
    
    if tracing_enabled and event_tracer:
        _replace_event_bus_with_traced(execution_container, "execution", event_tracer)
        
    topology['containers']['execution'] = execution_container
    
    # Create stateless components (only risk validators needed for replay)
    from .helpers.component_builder import create_stateless_components
    
    # Filter config to only include risk components
    filtered_config = {
        'risk_profiles': config.get('risk_profiles', [])
    }
    stateless_components = create_stateless_components(filtered_config)
    topology['stateless_components'] = stateless_components
    
    logger.info(f"Signal replay topology created: {len(topology['containers'])} containers, "
               f"{len(param_combos)} parameter combinations")
    
    # Wire containers together
    from .helpers.adapter_wiring import wire_signal_replay_topology
    
    # Add config info needed for wiring
    wiring_config = {
        'root_event_bus': root_event_bus,
        'stateless_components': topology['stateless_components'],
        'parameter_combinations': topology['parameter_combinations']
    }
    
    adapters = wire_signal_replay_topology(topology['containers'], wiring_config)
    topology['adapters'] = adapters
    
    return topology


# Import helper function from backtest topology
from .backtest import _replace_event_bus_with_traced