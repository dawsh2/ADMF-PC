"""
Enhanced dependency injection infrastructure for the execution module.

This module provides dependency injection patterns that align with the core
system's DI container and integrate properly with the Risk module's portfolio state.
"""

from typing import Protocol, Type, Any, Dict, Optional, List, Callable
from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from ..core.dependencies.container import DependencyContainer
from ..core.components.factory import ComponentFactory, create_component
from .protocols import (
    Broker,
    OrderProcessor,
    MarketSimulator,
    ExecutionEngine,
    ExecutionCapability,
)


class ExecutionDependencyProvider(Protocol):
    """Protocol for providing execution dependencies."""
    
    @abstractmethod
    def get_broker(self) -> Broker:
        """Get broker implementation."""
        ...
    
    @abstractmethod
    def get_order_manager(self) -> OrderProcessor:
        """Get order manager."""
        ...
    
    @abstractmethod
    def get_market_simulator(self) -> MarketSimulator:
        """Get market simulator."""
        ...
    
    @abstractmethod
    def get_execution_engine(self) -> ExecutionEngine:
        """Get execution engine."""
        ...


@dataclass
class ExecutionComponentSpec:
    """Specification for creating execution components."""
    component_type: str
    name: str
    class_name: str
    params: Dict[str, Any]
    dependencies: List[str]
    metadata: Dict[str, Any]


class ExecutionComponentFactory:
    """Factory for creating execution components with proper DI."""
    
    def __init__(self, dependency_container: DependencyContainer):
        """Initialize with dependency container."""
        self._container = dependency_container
        self._component_factory = ComponentFactory()
        
        # Register component types
        self._register_execution_components()
    
    def create_broker(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> Broker:
        """Create broker from specification."""
        return self._create_component(spec, context)
    
    def create_order_manager(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> OrderProcessor:
        """Create order manager from specification."""
        return self._create_component(spec, context)
    
    def create_market_simulator(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> MarketSimulator:
        """Create market simulator from specification."""
        return self._create_component(spec, context)
    
    def create_execution_engine(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEngine:
        """Create execution engine from specification."""
        return self._create_component(spec, context)
    
    def _create_component(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create component using factory with DI support."""
        # Get class from registry
        component_class = self._get_component_class(spec.class_name)
        
        # Resolve dependencies
        resolved_deps = {}
        for dep_name in spec.dependencies:
            if self._container.has(dep_name):
                resolved_deps[dep_name] = self._container.resolve(dep_name)
        
        # Merge context
        merged_context = {
            'container': self._container,
            'component_id': spec.name,
            **resolved_deps,
            **(context or {})
        }
        
        # Create with factory
        return self._component_factory.create(
            component_class,
            context=merged_context,
            **spec.params
        )
    
    def _get_component_class(self, class_name: str) -> Type[Any]:
        """Get component class by name."""
        # Import dynamically to avoid circular imports
        if class_name == 'BacktestBrokerRefactored':
            from .improved_backtest_broker import BacktestBrokerRefactored
            return BacktestBrokerRefactored
        elif class_name == 'OrderManager':
            from .improved_order_manager import ImprovedOrderManager
            return ImprovedOrderManager
        elif class_name == 'MarketSimulator':
            from .improved_market_simulation import ImprovedMarketSimulator
            return ImprovedMarketSimulator
        elif class_name == 'DefaultExecutionEngine':
            from .improved_execution_engine import ImprovedExecutionEngine
            return ImprovedExecutionEngine
        elif class_name == 'ExecutionContext':
            from .execution_context import ExecutionContext
            return ExecutionContext
        else:
            raise ValueError(f"Unknown component class: {class_name}")
    
    def _register_execution_components(self) -> None:
        """Register execution component types."""
        # Components are created on-demand, not registered globally
        pass


class ExecutionDependencyResolver:
    """Resolves dependencies for execution components."""
    
    def __init__(
        self,
        container: DependencyContainer,
        factory: ExecutionComponentFactory
    ):
        """Initialize with container and factory."""
        self._container = container
        self._factory = factory
        
        # Component registrations
        self._broker: Optional[Broker] = None
        self._order_manager: Optional[OrderProcessor] = None
        self._market_simulator: Optional[MarketSimulator] = None
        self._execution_engine: Optional[ExecutionEngine] = None
    
    def register_broker(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> Broker:
        """Register and create broker."""
        broker = self._factory.create_broker(spec, context)
        self._broker = broker
        
        # Register in DI container
        self._container.register_instance('Broker', broker)
        
        return broker
    
    def register_order_manager(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> OrderProcessor:
        """Register and create order manager."""
        order_manager = self._factory.create_order_manager(spec, context)
        self._order_manager = order_manager
        
        # Register in DI container
        self._container.register_instance('OrderProcessor', order_manager)
        
        return order_manager
    
    def register_market_simulator(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> MarketSimulator:
        """Register and create market simulator."""
        market_simulator = self._factory.create_market_simulator(spec, context)
        self._market_simulator = market_simulator
        
        # Register in DI container
        self._container.register_instance('MarketSimulator', market_simulator)
        
        return market_simulator
    
    def register_execution_engine(
        self,
        spec: ExecutionComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEngine:
        """Register and create execution engine."""
        execution_engine = self._factory.create_execution_engine(spec, context)
        self._execution_engine = execution_engine
        
        # Register in DI container
        self._container.register_instance('ExecutionEngine', execution_engine)
        
        return execution_engine
    
    def get_broker(self) -> Broker:
        """Get broker."""
        if not self._broker:
            raise ValueError("Broker not registered")
        return self._broker
    
    def get_order_manager(self) -> OrderProcessor:
        """Get order manager."""
        if not self._order_manager:
            raise ValueError("Order manager not registered")
        return self._order_manager
    
    def get_market_simulator(self) -> MarketSimulator:
        """Get market simulator."""
        if not self._market_simulator:
            raise ValueError("Market simulator not registered")
        return self._market_simulator
    
    def get_execution_engine(self) -> ExecutionEngine:
        """Get execution engine."""
        if not self._execution_engine:
            raise ValueError("Execution engine not registered")
        return self._execution_engine


def create_broker_spec(
    broker_type: str,
    name: str,
    **params
) -> ExecutionComponentSpec:
    """Create broker specification."""
    class_mapping = {
        'backtest': 'BacktestBrokerRefactored',
        'live': 'LiveBroker',  # Future implementation
        'paper': 'PaperBroker'  # Future implementation
    }
    
    class_name = class_mapping.get(broker_type)
    if not class_name:
        raise ValueError(f"Unknown broker type: {broker_type}")
    
    # Broker depends on portfolio state from Risk module
    dependencies = ['PortfolioState'] if broker_type == 'backtest' else []
    
    return ExecutionComponentSpec(
        component_type='broker',
        name=name,
        class_name=class_name,
        params=params,
        dependencies=dependencies,
        metadata={'type': broker_type}
    )


def create_order_manager_spec(
    name: str = 'order_manager',
    **params
) -> ExecutionComponentSpec:
    """Create order manager specification."""
    return ExecutionComponentSpec(
        component_type='order_manager',
        name=name,
        class_name='OrderManager',
        params=params,
        dependencies=[],
        metadata={'type': 'standard'}
    )


def create_market_simulator_spec(
    simulator_type: str = 'standard',
    name: str = 'market_simulator',
    **params
) -> ExecutionComponentSpec:
    """Create market simulator specification."""
    return ExecutionComponentSpec(
        component_type='market_simulator',
        name=name,
        class_name='MarketSimulator',
        params=params,
        dependencies=[],
        metadata={'type': simulator_type}
    )


def create_execution_engine_spec(
    engine_type: str = 'default',
    name: str = 'execution_engine',
    **params
) -> ExecutionComponentSpec:
    """Create execution engine specification."""
    class_mapping = {
        'default': 'DefaultExecutionEngine',
        'high_frequency': 'HighFrequencyExecutionEngine',  # Future
        'algorithmic': 'AlgorithmicExecutionEngine'  # Future
    }
    
    class_name = class_mapping.get(engine_type)
    if not class_name:
        raise ValueError(f"Unknown execution engine type: {engine_type}")
    
    # Execution engine depends on all other execution components
    dependencies = ['Broker', 'OrderProcessor', 'MarketSimulator', 'ExecutionContext']
    
    return ExecutionComponentSpec(
        component_type='execution_engine',
        name=name,
        class_name=class_name,
        params=params,
        dependencies=dependencies,
        metadata={'type': engine_type}
    )


class ExecutionModuleBuilder:
    """Builder for complete execution module with proper DI."""
    
    def __init__(self, dependency_container: DependencyContainer):
        """Initialize builder with DI container."""
        self._container = dependency_container
        self._factory = ExecutionComponentFactory(dependency_container)
        self._resolver = ExecutionDependencyResolver(dependency_container, self._factory)
    
    def with_broker(
        self,
        broker_type: str,
        **params
    ) -> 'ExecutionModuleBuilder':
        """Configure broker."""
        spec = create_broker_spec(broker_type, 'broker', **params)
        self._resolver.register_broker(spec)
        return self
    
    def with_order_manager(self, **params) -> 'ExecutionModuleBuilder':
        """Configure order manager."""
        spec = create_order_manager_spec('order_manager', **params)
        self._resolver.register_order_manager(spec)
        return self
    
    def with_market_simulator(
        self,
        simulator_type: str = 'standard',
        **params
    ) -> 'ExecutionModuleBuilder':
        """Configure market simulator."""
        spec = create_market_simulator_spec(simulator_type, 'market_simulator', **params)
        self._resolver.register_market_simulator(spec)
        return self
    
    def with_execution_engine(
        self,
        engine_type: str = 'default',
        **params
    ) -> 'ExecutionModuleBuilder':
        """Configure execution engine."""
        spec = create_execution_engine_spec(engine_type, 'execution_engine', **params)
        self._resolver.register_execution_engine(spec)
        return self
    
    def build(self) -> ExecutionDependencyProvider:
        """Build complete execution module."""
        # Ensure all components are registered
        if not self._resolver._broker:
            self.with_broker('backtest')
        if not self._resolver._order_manager:
            self.with_order_manager()
        if not self._resolver._market_simulator:
            self.with_market_simulator()
        if not self._resolver._execution_engine:
            self.with_execution_engine()
        
        return self._resolver


def create_execution_module(
    dependency_container: DependencyContainer,
    config: Dict[str, Any]
) -> ExecutionDependencyProvider:
    """Factory function to create complete execution module."""
    builder = ExecutionModuleBuilder(dependency_container)
    
    # Configure from config
    broker_config = config.get('broker', {})
    builder.with_broker(
        broker_type=broker_config.get('type', 'backtest'),
        **broker_config.get('params', {})
    )
    
    order_config = config.get('order_manager', {})
    builder.with_order_manager(**order_config.get('params', {}))
    
    simulator_config = config.get('market_simulator', {})
    builder.with_market_simulator(
        simulator_type=simulator_config.get('type', 'standard'),
        **simulator_config.get('params', {})
    )
    
    engine_config = config.get('execution_engine', {})
    builder.with_execution_engine(
        engine_type=engine_config.get('type', 'default'),
        **engine_config.get('params', {})
    )
    
    return builder.build()
