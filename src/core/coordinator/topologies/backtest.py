"""
Backtest topology: Full pipeline execution.

Pipeline: data → features → strategies → portfolios → risk → execution
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from ...containers.container import Container, ContainerConfig, ContainerRole
from ...containers.components import (
    DataStreamer,
    FeatureCalculator,
    PortfolioState,
    SignalProcessor,
    OrderGenerator,
    ExecutionEngine
)
from ...events.tracing import TracedEventBus, EventTracer
from ...events import EventBus, EventType

logger = logging.getLogger(__name__)


def create_backtest_topology(config: Dict[str, Any], tracing_enabled: bool = True) -> Dict[str, Any]:
    """
    Create topology for full backtest execution.
    
    Creates:
    - Symbol-Timeframe containers for data and features
    - Portfolio containers for each parameter combination
    - Execution container for order processing
    - Stateless strategy/classifier/risk components
    
    Returns:
        Dictionary with containers, components, and metadata
    """
    # Create root event bus for inter-component communication
    if tracing_enabled:
        root_event_bus = TracedEventBus("root_event_bus")
        correlation_id = config.get('correlation_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        event_tracer = EventTracer(
            correlation_id=correlation_id,
            max_events=config.get('tracing', {}).get('max_events', 10000)
        )
        root_event_bus.set_tracer(event_tracer)
        logger.info("🔍 Created TracedEventBus with event tracing enabled")
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
    
    # Infer required features from strategies
    from ....strategy.components.feature_inference import infer_features_from_strategies
    strategies = config.get('strategies', [])
    inferred_features = infer_features_from_strategies(strategies)
    logger.info(f"Inferred features from strategies: {sorted(inferred_features)}")
    
    # Add inferred features to each symbol config
    for st_config in symbol_timeframe_configs:
        if 'features' not in st_config:
            st_config['features'] = {}
        if 'indicators' not in st_config['features']:
            st_config['features']['indicators'] = []
            
        # Convert inferred features to indicator configs
        for feature_spec in inferred_features:
            if '_' in feature_spec:
                parts = feature_spec.split('_')
                feature_type = parts[0]
                if len(parts) > 1 and parts[1].isdigit():
                    period = int(parts[1])
                    st_config['features']['indicators'].append({
                        'name': feature_spec,
                        'type': feature_type,
                        'period': period
                    })
                else:
                    st_config['features']['indicators'].append({
                        'name': feature_spec,
                        'type': feature_type
                    })
    
    # Create Symbol-Timeframe containers
    for st_config in symbol_timeframe_configs:
        symbol = st_config['symbol']
        timeframe = st_config.get('timeframe', '1d')
        
        # Data container - just streams data
        data_container_id = f"{symbol}_{timeframe}_data"
        data_container = Container(ContainerConfig(
            role=ContainerRole.DATA,
            name=data_container_id,
            container_id=data_container_id
        ))
        
        # Add data streaming component
        data_container.add_component('data_streamer', DataStreamer(
            symbol=symbol,
            timeframe=timeframe,
            data_source=st_config.get('data_config', {})
        ))
        
        if tracing_enabled and event_tracer:
            _replace_event_bus_with_traced(data_container, data_container_id, event_tracer)
            
        topology['containers'][data_container_id] = data_container
        
        # Feature container - just calculates features
        feature_container_id = f"{symbol}_{timeframe}_features"
        feature_container = Container(ContainerConfig(
            role=ContainerRole.FEATURE,
            name=feature_container_id,
            container_id=feature_container_id
        ))
        
        # Add feature calculation component
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
        
        # Wire data streamer to feature calculator
        feature_calc = feature_container.get_component('feature_calculator')
        if feature_calc and hasattr(feature_calc, 'on_bar'):
            data_container.event_bus.subscribe('BAR', feature_calc.on_bar)
        logger.info(f"Wired {data_container_id} → {feature_container_id}")
    
    # Create stateless components
    topology['stateless_components'] = _create_stateless_components(config)
    
    # Expand parameter combinations
    param_combos = _expand_parameter_combinations(config)
    topology['parameter_combinations'] = param_combos
    
    # Create Portfolio containers
    for combo in param_combos:
        combo_id = combo['combo_id']
        
        portfolio_container = Container(ContainerConfig(
            role=ContainerRole.PORTFOLIO,
            name=f'portfolio_{combo_id}',
            container_id=f'portfolio_{combo_id}'
        ))
        
        # Add portfolio components
        portfolio_container.add_component('portfolio_state', PortfolioState(
            initial_capital=config.get('portfolio', {}).get('initial_capital', 100000)
        ))
        
        portfolio_container.add_component('signal_processor', SignalProcessor())
        portfolio_container.add_component('order_generator', OrderGenerator())
        
        if tracing_enabled and event_tracer:
            _replace_event_bus_with_traced(portfolio_container, f"portfolio_{combo_id}", event_tracer)
            
        # Give portfolio access to root event bus
        portfolio_container.root_event_bus = root_event_bus
        
        topology['containers'][f'portfolio_{combo_id}'] = portfolio_container
    
    # Create Execution container
    execution_container = Container(ContainerConfig(
        role=ContainerRole.EXECUTION,
        name='execution',
        container_id='execution'
    ))
    
    # Add execution engine component
    exec_config = config.get('execution', {})
    execution_container.add_component('execution_engine', ExecutionEngine(
        slippage_model=exec_config.get('slippage_model'),
        commission_model=exec_config.get('commission_model')
    ))
    
    if tracing_enabled and event_tracer:
        _replace_event_bus_with_traced(execution_container, "execution", event_tracer)
        
    topology['containers']['execution'] = execution_container
    
    logger.info(f"Backtest topology created: {len(topology['containers'])} containers, "
               f"{len(param_combos)} parameter combinations")
    
    # Wire containers together
    from .helpers.adapter_wiring import wire_backtest_topology
    
    # Add config info needed for wiring
    wiring_config = {
        'root_event_bus': root_event_bus,
        'stateless_components': topology['stateless_components'],
        'parameter_combinations': topology['parameter_combinations']
    }
    
    adapters = wire_backtest_topology(topology['containers'], wiring_config)
    topology['adapters'] = adapters
    
    return topology


def _extract_symbol_timeframe_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract symbol-timeframe configurations from config."""
    symbol_configs = []
    
    # Simple symbols list
    if 'symbols' in config:
        symbols = config['symbols']
        if isinstance(symbols, list) and all(isinstance(s, str) for s in symbols):
            for symbol in symbols:
                symbol_configs.append({
                    'symbol': symbol,
                    'timeframe': '1d',
                    'data_config': config.get('data', {}),
                    'features': config.get('features', {})
                })
            return symbol_configs
    
    # Detailed symbol_configs
    if 'symbol_configs' in config:
        for sc in config['symbol_configs']:
            symbol = sc['symbol']
            timeframes = sc.get('timeframes', ['1d'])
            if not isinstance(timeframes, list):
                timeframes = [timeframes]
            
            for timeframe in timeframes:
                symbol_configs.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_config': sc.get('data_config', config.get('data', {})),
                    'features': sc.get('features', config.get('features', {}))
                })
    
    # Default case
    if not symbol_configs:
        logger.warning("No symbol configurations found, using default SPY_1d")
        symbol_configs.append({
            'symbol': 'SPY',
            'timeframe': '1d',
            'data_config': config.get('data', {}),
            'features': config.get('features', {})
        })
    
    return symbol_configs


def _expand_parameter_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand parameter grid into individual combinations."""
    combinations = []
    
    strategy_params = config.get('strategies', [{}])
    risk_params = config.get('risk_profiles', [{}])
    classifier_params = config.get('classifiers', [{}])
    execution_params = config.get('execution_models', [{}])
    
    # Ensure they're lists
    if not isinstance(strategy_params, list):
        strategy_params = [strategy_params]
    if not isinstance(risk_params, list):
        risk_params = [risk_params]
    if not isinstance(classifier_params, list):
        classifier_params = [classifier_params]
    if not isinstance(execution_params, list):
        execution_params = [execution_params]
    
    # Generate all combinations
    combo_id = 0
    for strat in strategy_params:
        for risk in risk_params:
            for classifier in classifier_params:
                for execution in execution_params:
                    combinations.append({
                        'combo_id': f'c{combo_id:04d}',
                        'strategy_params': strat,
                        'risk_params': risk,
                        'classifier_params': classifier,
                        'execution_params': execution
                    })
                    combo_id += 1
    
    return combinations


def _create_stateless_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create stateless strategy, classifier, risk, and execution components."""
    # This will be imported from helpers
    from .helpers.component_builder import create_stateless_components
    return create_stateless_components(config)


def _replace_event_bus_with_traced(container, container_id: str, event_tracer):
    """Replace container's event bus with traced version."""
    traced_bus = TracedEventBus(f"{container_id}_bus")
    traced_bus.set_tracer(event_tracer)
    
    # Copy existing subscriptions
    if hasattr(container.event_bus, '_subscribers'):
        traced_bus._subscribers = container.event_bus._subscribers
        traced_bus._handler_refs = container.event_bus._handler_refs
        
    container.event_bus = traced_bus