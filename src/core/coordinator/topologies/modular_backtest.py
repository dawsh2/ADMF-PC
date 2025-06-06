"""
Modular backtest topology demonstrating component-based container composition.

This topology shows how to build a backtest pipeline using:
- Generic containers with specific components
- Clear separation of concerns (data, features, strategies, etc.)
- Flexible pipeline that can be easily modified
"""

from typing import Dict, Any, List
import logging

from ...containers.container import Container, ContainerConfig, ContainerRole
from ...containers.components import (
    DataStreamer,
    FeatureCalculator,
    PortfolioState,
    SignalProcessor,
    OrderGenerator,
    RiskValidator,
    ExecutionEngine
)
from ...communication.factory import AdapterFactory

logger = logging.getLogger(__name__)


def create_backtest_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a modular backtest topology.
    
    This demonstrates the ideal architecture where:
    - Containers are generic vessels
    - Components provide specific functionality
    - Pipeline is defined by topology, not hardcoded
    
    Args:
        config: Configuration with symbols, strategies, features, etc.
        
    Returns:
        Dictionary with containers and adapters
    """
    containers = {}
    
    # Extract configuration
    symbols = config.get('symbols', [])
    features_config = config.get('features', {})
    strategies = config.get('strategies', [])
    risk_profiles = config.get('risk_profiles', [])
    execution_config = config.get('execution', {})
    
    # 1. Create Symbol-specific containers
    for symbol_config in symbols:
        symbol = symbol_config['symbol']
        timeframes = symbol_config.get('timeframes', ['1d'])
        
        for timeframe in timeframes:
            # Data container - just streams bars
            data_container_id = f"{symbol}_{timeframe}_data"
            data_container = Container(ContainerConfig(
                role=ContainerRole.DATA,
                name=data_container_id,
                container_id=data_container_id
            ))
            
            # Add data streaming component
            data_streamer = DataStreamer(
                symbol=symbol,
                timeframe=timeframe,
                data_source=symbol_config.get('data_source')
            )
            data_container.add_component(data_streamer)
            
            containers[data_container_id] = data_container
            
            # Feature container - calculates indicators
            feature_container_id = f"{symbol}_{timeframe}_features"
            feature_container = Container(ContainerConfig(
                role=ContainerRole.FEATURE,
                name=feature_container_id,
                container_id=feature_container_id
            ))
            
            # Add feature calculation component
            feature_calculator = FeatureCalculator(
                indicators=features_config.get('indicators', []),
                lookback_window=features_config.get('lookback_window', 100)
            )
            feature_container.add_component(feature_calculator)
            
            containers[feature_container_id] = feature_container
    
    # 2. Create Strategy containers (stateless services)
    # In the modular approach, strategies are stateless services that process features
    # They don't need their own containers - they're called by portfolios
    
    # 3. Create Portfolio containers (one per parameter combination)
    portfolio_id = 0
    for strategy_config in strategies:
        for risk_profile in risk_profiles:
            portfolio_container_id = f"portfolio_{portfolio_id:04d}"
            portfolio_container = Container(ContainerConfig(
                role=ContainerRole.PORTFOLIO,
                name=portfolio_container_id,
                container_id=portfolio_container_id
            ))
            
            # Add portfolio components
            portfolio_state = PortfolioState(
                initial_capital=config.get('initial_capital', 100000)
            )
            portfolio_container.add_component(portfolio_state)
            
            signal_processor = SignalProcessor()
            portfolio_container.add_component(signal_processor)
            
            order_generator = OrderGenerator()
            portfolio_container.add_component(order_generator)
            
            containers[portfolio_container_id] = portfolio_container
            portfolio_id += 1
    
    # 4. Create Risk container (optional - could be part of portfolio)
    if config.get('separate_risk_container', False):
        risk_container = Container(ContainerConfig(
            role=ContainerRole.RISK,
            name="risk_manager",
            container_id="risk_manager"
        ))
        
        risk_validator = RiskValidator(
            max_position_size=config.get('max_position_size', 0.1),
            max_portfolio_risk=config.get('max_portfolio_risk', 0.02)
        )
        risk_container.add_component(risk_validator)
        
        containers["risk_manager"] = risk_container
    
    # 5. Create Execution container
    execution_container = Container(ContainerConfig(
        role=ContainerRole.EXECUTION,
        name="execution_engine",
        container_id="execution_engine"
    ))
    
    execution_engine = ExecutionEngine(
        slippage_model=execution_config.get('slippage_model'),
        commission_model=execution_config.get('commission_model')
    )
    execution_container.add_component(execution_engine)
    
    containers["execution_engine"] = execution_container
    
    # 6. Create adapters to wire the pipeline
    adapters = create_pipeline_adapters(containers, config)
    
    return {
        'containers': containers,
        'adapters': adapters,
        'topology_type': 'modular_backtest'
    }


def create_pipeline_adapters(containers: Dict[str, Container], config: Dict[str, Any]) -> List[Any]:
    """
    Create adapters to wire containers into a pipeline.
    
    This is where the magic happens - we define the data flow without
    hardcoding it into container types.
    
    The pipeline:
    1. Data containers → Feature containers (BAR events)
    2. Feature containers → Strategies → Portfolios (FEATURES → SIGNALS)
    3. Portfolios → Risk → Execution (ORDER_REQUEST → ORDER)
    4. Execution → Portfolios (FILL events)
    """
    adapter_factory = AdapterFactory()
    adapters = []
    
    # 1. Wire Data → Features for each symbol-timeframe
    data_containers = {k: v for k, v in containers.items() if '_data' in k}
    feature_containers = {k: v for k, v in containers.items() if '_features' in k}
    
    for data_id, data_container in data_containers.items():
        # Find matching feature container
        symbol_timeframe = data_id.replace('_data', '')
        feature_id = f"{symbol_timeframe}_features"
        
        if feature_id in feature_containers:
            # Create adapter: data.BAR → features.on_bar
            adapter = adapter_factory.create_adapter(
                name=f"data_to_features_{symbol_timeframe}",
                config={
                    'type': 'direct',
                    'source': data_id,
                    'target': feature_id,
                    'event_mappings': {
                        'BAR': 'on_bar'
                    }
                }
            )
            adapters.append(adapter)
    
    # 2. Wire Features → Strategies → Portfolios
    # This is more complex as strategies are stateless services
    # We'll use a FeatureDispatcher pattern
    
    portfolio_containers = {k: v for k, v in containers.items() if 'portfolio_' in k}
    
    # Create feature dispatcher adapter
    feature_dispatcher = adapter_factory.create_adapter(
        name="feature_dispatcher",
        config={
            'type': 'feature_dispatcher',
            'sources': list(feature_containers.keys()),
            'targets': list(portfolio_containers.keys()),
            'strategy_services': config.get('strategies', [])
        }
    )
    adapters.append(feature_dispatcher)
    
    # 3. Wire Portfolios → Risk → Execution
    if 'risk_manager' in containers:
        # Portfolios → Risk
        risk_adapter = adapter_factory.create_adapter(
            name="portfolio_to_risk",
            config={
                'type': 'aggregator',
                'sources': list(portfolio_containers.keys()),
                'target': 'risk_manager',
                'event_types': ['ORDER_REQUEST']
            }
        )
        adapters.append(risk_adapter)
        
        # Risk → Execution
        risk_to_exec = adapter_factory.create_adapter(
            name="risk_to_execution",
            config={
                'type': 'direct',
                'source': 'risk_manager',
                'target': 'execution_engine',
                'event_mappings': {
                    'ORDER': 'on_order'
                }
            }
        )
        adapters.append(risk_to_exec)
    else:
        # Direct: Portfolios → Execution
        portfolio_to_exec = adapter_factory.create_adapter(
            name="portfolio_to_execution",
            config={
                'type': 'aggregator',
                'sources': list(portfolio_containers.keys()),
                'target': 'execution_engine',
                'event_types': ['ORDER']
            }
        )
        adapters.append(portfolio_to_exec)
    
    # 4. Wire Execution → Portfolios (broadcast fills)
    fill_broadcast = adapter_factory.create_adapter(
        name="fill_broadcast",
        config={
            'type': 'broadcast',
            'source': 'execution_engine',
            'targets': list(portfolio_containers.keys()),
            'event_types': ['FILL']
        }
    )
    adapters.append(fill_broadcast)
    
    return adapters


def modify_pipeline_order(topology: Dict[str, Any], new_order: List[str]) -> Dict[str, Any]:
    """
    Demonstrate how easy it is to modify pipeline order with modular approach.
    
    For example, to put risk BEFORE portfolio instead of after:
    new_order = ['data', 'features', 'strategies', 'risk', 'portfolios', 'execution']
    
    This would be very difficult with hardcoded container types!
    """
    # Re-wire adapters based on new order
    # Implementation left as exercise
    pass