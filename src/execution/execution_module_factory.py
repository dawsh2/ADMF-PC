"""
Execution module factory that integrates with the core DI system.

This factory creates complete execution modules with proper dependency
injection, following the patterns established by the Risk module.
"""

from typing import Dict, Any, Optional, Type, List
from dataclasses import dataclass
from decimal import Decimal
import logging

from ..core.dependencies.container import DependencyContainer
from ..core.components.factory import ComponentFactory
from ..risk.protocols import PortfolioStateProtocol
from .protocols import (
    Broker, OrderProcessor, MarketSimulator, ExecutionEngine,
    ExecutionCapability
)
from .improved_execution_engine import ImprovedExecutionEngine
from .improved_backtest_broker import BacktestBrokerRefactored, create_backtest_broker
from .improved_order_manager import ImprovedOrderManager, create_order_manager
from .improved_market_simulation import (
    ImprovedMarketSimulator, create_market_simulator,
    create_conservative_simulator, create_realistic_simulator
)
from .execution_context import ExecutionContext

logger = logging.getLogger(__name__)


@dataclass
class ExecutionModuleConfig:
    """Configuration for execution module creation."""
    
    # Broker configuration
    broker_type: str = "backtest"
    broker_params: Dict[str, Any] = None
    
    # Order manager configuration
    order_manager_params: Dict[str, Any] = None
    
    # Market simulator configuration
    simulator_type: str = "realistic"
    simulator_params: Dict[str, Any] = None
    
    # Execution engine configuration
    engine_type: str = "improved"
    engine_params: Dict[str, Any] = None
    
    # Context configuration
    context_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default parameters."""
        if self.broker_params is None:
            self.broker_params = {}
        if self.order_manager_params is None:
            self.order_manager_params = {}
        if self.simulator_params is None:
            self.simulator_params = {}
        if self.engine_params is None:
            self.engine_params = {}
        if self.context_params is None:
            self.context_params = {}


class ExecutionModuleFactory:
    """Factory for creating complete execution modules with proper DI."""
    
    def __init__(self, dependency_container: DependencyContainer):
        """Initialize with dependency container.
        
        Args:
            dependency_container: Core system's dependency container
        """
        self._container = dependency_container
        self._component_factory = ComponentFactory()
        
        # Register execution component types with the core factory
        self._register_execution_components()
        
        logger.info("ExecutionModuleFactory initialized")
    
    def create_execution_module(
        self,
        config: ExecutionModuleConfig,
        portfolio_state: PortfolioStateProtocol,
        module_id: str = "execution"
    ) -> Dict[str, Any]:
        """Create complete execution module with all components.
        
        Args:
            config: Execution module configuration
            portfolio_state: Portfolio state from Risk module (required dependency)
            module_id: Unique module identifier
            
        Returns:
            Dictionary with all execution components
        """
        logger.info(f"Creating execution module - ID: {module_id}")
        
        # Create execution context first (no dependencies)
        context = self._create_execution_context(f"{module_id}_context", config.context_params)
        
        # Create order manager (no external dependencies)
        order_manager = self._create_order_manager(f"{module_id}_order_mgr", config.order_manager_params)
        
        # Create market simulator (no external dependencies)
        market_simulator = self._create_market_simulator(f"{module_id}_sim", config)
        
        # Create broker (depends on portfolio state)
        broker = self._create_broker(f"{module_id}_broker", config, portfolio_state)
        
        # Create execution engine (depends on all other components)
        execution_engine = self._create_execution_engine(
            f"{module_id}_engine",
            config,
            broker,
            order_manager,
            market_simulator,
            context
        )
        
        # Register components in DI container
        self._register_components_in_container(
            module_id,
            {
                'execution_context': context,
                'order_manager': order_manager,
                'market_simulator': market_simulator,
                'broker': broker,
                'execution_engine': execution_engine
            }
        )
        
        # Initialize all components
        self._initialize_components(
            [context, order_manager, market_simulator, broker, execution_engine]
        )
        
        logger.info(f"Execution module created successfully - ID: {module_id}")
        
        return {
            'execution_engine': execution_engine,
            'broker': broker,
            'order_manager': order_manager,
            'market_simulator': market_simulator,
            'execution_context': context,
            'module_id': module_id
        }
    
    def create_backtest_execution_module(
        self,
        portfolio_state: PortfolioStateProtocol,
        module_id: str = "backtest_execution",
        conservative: bool = False
    ) -> Dict[str, Any]:
        """Create execution module optimized for backtesting.
        
        Args:
            portfolio_state: Portfolio state from Risk module
            module_id: Unique module identifier
            conservative: Whether to use conservative simulation models
            
        Returns:
            Dictionary with execution components optimized for backtesting
        """
        # Configure for backtesting
        config = ExecutionModuleConfig(
            broker_type="backtest",
            broker_params={
                'commission_rate': 0.001 if not conservative else 0.002,
                'slippage_rate': 0.0005 if not conservative else 0.001
            },
            simulator_type="conservative" if conservative else "realistic",
            order_manager_params={
                'max_order_age_hours': 24,
                'validation_enabled': True
            },
            engine_params={
                'validate_orders': True,
                'enable_metrics': True
            }
        )
        
        return self.create_execution_module(config, portfolio_state, module_id)
    
    def create_live_execution_module(
        self,
        portfolio_state: PortfolioStateProtocol,
        broker_config: Dict[str, Any],
        module_id: str = "live_execution"
    ) -> Dict[str, Any]:
        """Create execution module for live trading.
        
        Args:
            portfolio_state: Portfolio state from Risk module
            broker_config: Live broker configuration
            module_id: Unique module identifier
            
        Returns:
            Dictionary with execution components for live trading
        """
        # Configure for live trading
        config = ExecutionModuleConfig(
            broker_type="live",
            broker_params=broker_config,
            simulator_type="none",  # No simulation in live trading
            order_manager_params={
                'max_order_age_hours': 1,  # Shorter age for live orders
                'validation_enabled': True
            },
            engine_params={
                'validate_orders': True,
                'enable_metrics': True,
                'live_mode': True
            }
        )
        
        return self.create_execution_module(config, portfolio_state, module_id)
    
    # Private creation methods
    
    def _create_execution_context(
        self,
        component_id: str,
        params: Dict[str, Any]
    ) -> ExecutionContext:
        """Create execution context."""
        return ExecutionContext()
    
    def _create_order_manager(
        self,
        component_id: str,
        params: Dict[str, Any]
    ) -> ImprovedOrderManager:
        """Create order manager."""
        return create_order_manager(
            component_id=component_id,
            **params
        )
    
    def _create_market_simulator(
        self,
        component_id: str,
        config: ExecutionModuleConfig
    ) -> ImprovedMarketSimulator:
        """Create market simulator."""
        if config.simulator_type == "conservative":
            return create_conservative_simulator(component_id)
        elif config.simulator_type == "realistic":
            return create_realistic_simulator(component_id)
        elif config.simulator_type == "custom":
            return create_market_simulator(
                component_id=component_id,
                **config.simulator_params
            )
        elif config.simulator_type == "none":
            # Return a null simulator for live trading
            return None
        else:
            raise ValueError(f"Unknown simulator type: {config.simulator_type}")
    
    def _create_broker(
        self,
        component_id: str,
        config: ExecutionModuleConfig,
        portfolio_state: PortfolioStateProtocol
    ) -> Broker:
        """Create broker with portfolio state dependency."""
        if config.broker_type == "backtest":
            return create_backtest_broker(
                component_id=component_id,
                portfolio_state=portfolio_state,
                **config.broker_params
            )
        elif config.broker_type == "live":
            # Future implementation for live brokers
            raise NotImplementedError("Live brokers not yet implemented")
        elif config.broker_type == "paper":
            # Future implementation for paper trading
            raise NotImplementedError("Paper trading brokers not yet implemented")
        else:
            raise ValueError(f"Unknown broker type: {config.broker_type}")
    
    def _create_execution_engine(
        self,
        component_id: str,
        config: ExecutionModuleConfig,
        broker: Broker,
        order_manager: OrderProcessor,
        market_simulator: Optional[MarketSimulator],
        context: ExecutionContext
    ) -> ImprovedExecutionEngine:
        """Create execution engine with all dependencies."""
        if config.engine_type == "improved":
            return ImprovedExecutionEngine(
                component_id=component_id,
                broker=broker,
                order_manager=order_manager,
                market_simulator=market_simulator,
                execution_context=context
            )
        else:
            raise ValueError(f"Unknown execution engine type: {config.engine_type}")
    
    def _register_components_in_container(
        self,
        module_id: str,
        components: Dict[str, Any]
    ) -> None:
        """Register all components in the DI container."""
        for component_name, component in components.items():
            registration_key = f"{module_id}_{component_name}"
            
            # Register by interface type
            if hasattr(component, '__class__'):
                interface_name = component.__class__.__bases__[0].__name__ if component.__class__.__bases__ else component.__class__.__name__
                self._container.register_instance(interface_name, component)
            
            # Also register by specific key
            self._container.register_instance(registration_key, component)
            
            logger.debug(f"Registered component: {registration_key}")
    
    def _initialize_components(self, components: List[Any]) -> None:
        """Initialize all components in dependency order."""
        context = {'container': self._container}
        
        for component in components:
            if hasattr(component, 'initialize'):
                try:
                    component.initialize(context)
                    logger.debug(f"Initialized component: {component.component_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize component {component.component_id}: {e}")
                    raise
    
    def _register_execution_components(self) -> None:
        """Register execution component types with the core factory."""
        # Register component creation functions
        self._component_factory.register(
            'ExecutionContext',
            lambda **kwargs: ExecutionContext()
        )
        
        self._component_factory.register(
            'ImprovedOrderManager',
            lambda component_id, **kwargs: create_order_manager(component_id, **kwargs)
        )
        
        self._component_factory.register(
            'ImprovedMarketSimulator',
            lambda component_id, simulator_type='realistic', **kwargs: (
                create_conservative_simulator(component_id) if simulator_type == 'conservative'
                else create_realistic_simulator(component_id) if simulator_type == 'realistic'
                else create_market_simulator(component_id, **kwargs)
            )
        )
        
        self._component_factory.register(
            'BacktestBrokerRefactored',
            lambda component_id, portfolio_state, **kwargs: create_backtest_broker(
                component_id, portfolio_state, **kwargs
            )
        )
        
        self._component_factory.register(
            'ImprovedExecutionEngine',
            lambda component_id, broker, order_manager, market_simulator, execution_context, **kwargs: (
                ImprovedExecutionEngine(
                    component_id, broker, order_manager, market_simulator, execution_context
                )
            )
        )


class ExecutionContainerBootstrap:
    """Bootstrap utility for creating execution modules in container environments."""
    
    def __init__(self, dependency_container: DependencyContainer):
        """Initialize bootstrap with dependency container."""
        self._container = dependency_container
        self._factory = ExecutionModuleFactory(dependency_container)
    
    def bootstrap_backtest_execution(
        self,
        portfolio_state_key: str = 'PortfolioState',
        module_id: str = 'execution',
        conservative: bool = False
    ) -> Dict[str, Any]:
        """Bootstrap execution module for backtesting.
        
        Args:
            portfolio_state_key: Key to resolve portfolio state from container
            module_id: Module identifier
            conservative: Whether to use conservative settings
            
        Returns:
            Dictionary with execution components
        """
        # Resolve portfolio state from container
        if not self._container.has(portfolio_state_key):
            raise ValueError(f"Portfolio state not found in container: {portfolio_state_key}")
        
        portfolio_state = self._container.resolve(portfolio_state_key)
        
        # Create execution module
        return self._factory.create_backtest_execution_module(
            portfolio_state=portfolio_state,
            module_id=module_id,
            conservative=conservative
        )
    
    def bootstrap_from_config(
        self,
        config: Dict[str, Any],
        portfolio_state_key: str = 'PortfolioState'
    ) -> Dict[str, Any]:
        """Bootstrap execution module from configuration.
        
        Args:
            config: Configuration dictionary
            portfolio_state_key: Key to resolve portfolio state from container
            
        Returns:
            Dictionary with execution components
        """
        # Resolve portfolio state
        portfolio_state = self._container.resolve(portfolio_state_key)
        
        # Create execution module config
        execution_config = ExecutionModuleConfig(
            broker_type=config.get('broker_type', 'backtest'),
            broker_params=config.get('broker_params', {}),
            order_manager_params=config.get('order_manager_params', {}),
            simulator_type=config.get('simulator_type', 'realistic'),
            simulator_params=config.get('simulator_params', {}),
            engine_type=config.get('engine_type', 'improved'),
            engine_params=config.get('engine_params', {}),
            context_params=config.get('context_params', {})
        )
        
        return self._factory.create_execution_module(
            config=execution_config,
            portfolio_state=portfolio_state,
            module_id=config.get('module_id', 'execution')
        )


def create_execution_module_factory(
    dependency_container: DependencyContainer
) -> ExecutionModuleFactory:
    """Factory function to create execution module factory."""
    return ExecutionModuleFactory(dependency_container)


def create_execution_bootstrap(
    dependency_container: DependencyContainer
) -> ExecutionContainerBootstrap:
    """Factory function to create execution bootstrap utility."""
    return ExecutionContainerBootstrap(dependency_container)


# Configuration builders for common scenarios

def build_conservative_backtest_config(
    commission_rate: float = 0.002,
    slippage_rate: float = 0.001,
    module_id: str = 'conservative_execution'
) -> ExecutionModuleConfig:
    """Build configuration for conservative backtesting."""
    return ExecutionModuleConfig(
        broker_type='backtest',
        broker_params={
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate
        },
        simulator_type='conservative',
        order_manager_params={
            'max_order_age_hours': 24,
            'validation_enabled': True
        },
        engine_params={
            'validate_orders': True,
            'enable_metrics': True
        }
    )


def build_realistic_backtest_config(
    commission_tiers: Optional[List[tuple]] = None,
    slippage_model: str = 'volume_impact',
    module_id: str = 'realistic_execution'
) -> ExecutionModuleConfig:
    """Build configuration for realistic backtesting."""
    if commission_tiers is None:
        commission_tiers = [
            (0, 0.003),      # $0-1k: 0.3%
            (1000, 0.002),   # $1k-10k: 0.2%
            (10000, 0.001)   # $10k+: 0.1%
        ]
    
    return ExecutionModuleConfig(
        broker_type='backtest',
        broker_params={
            'commission_rate': 0.001,  # Base rate
            'slippage_rate': 0.0005
        },
        simulator_type='realistic',
        simulator_params={
            'slippage_model': slippage_model,
            'commission_model': 'tiered',
            'commission_params': {
                'tiers': [(Decimal(str(threshold)), Decimal(str(rate))) 
                         for threshold, rate in commission_tiers]
            }
        },
        order_manager_params={
            'max_order_age_hours': 12,
            'validation_enabled': True
        },
        engine_params={
            'validate_orders': True,
            'enable_metrics': True
        }
    )


def build_high_frequency_config(
    ultra_low_latency: bool = True,
    module_id: str = 'hf_execution'
) -> ExecutionModuleConfig:
    """Build configuration for high-frequency trading simulation."""
    return ExecutionModuleConfig(
        broker_type='backtest',
        broker_params={
            'commission_rate': 0.0005,  # Lower commission for HF
            'slippage_rate': 0.0001     # Minimal slippage
        },
        simulator_type='custom',
        simulator_params={
            'slippage_model': 'percentage',
            'commission_model': 'per_share',
            'slippage_params': {
                'base_slippage_pct': Decimal('0.0001')  # 0.01%
            },
            'commission_params': {
                'rate_per_share': Decimal('0.001'),     # $0.001/share
                'minimum_commission': Decimal('0.1')    # $0.10 minimum
            },
            'simulator_params': {
                'fill_probability': Decimal('0.99'),
                'partial_fill_enabled': False,
                'max_participation_rate': Decimal('0.05')  # Low impact
            }
        },
        order_manager_params={
            'max_order_age_hours': 1,  # Very short lived orders
            'validation_enabled': True
        },
        engine_params={
            'validate_orders': True,
            'enable_metrics': True,
            'high_frequency_mode': True
        }
    )


# Integration helpers

def integrate_execution_with_risk(
    execution_module: Dict[str, Any],
    risk_container: Any,
    event_bus: Any
) -> None:
    """Integrate execution module with risk container and event bus.
    
    Args:
        execution_module: Execution module components
        risk_container: Risk container instance
        event_bus: System event bus
    """
    execution_engine = execution_module['execution_engine']
    broker = execution_module['broker']
    
    # Set event bus for execution engine
    if hasattr(execution_engine, 'event_bus'):
        execution_engine.event_bus = event_bus
    
    # Subscribe to risk events if needed
    if hasattr(risk_container, 'get_portfolio_state'):
        portfolio_state = risk_container.get_portfolio_state()
        
        # Ensure broker is using the same portfolio state
        if hasattr(broker, '_portfolio_state'):
            assert broker._portfolio_state is portfolio_state, \
                "Broker must use the same portfolio state as risk container"
    
    logger.info("Execution module integrated with risk container")


def validate_execution_module(
    execution_module: Dict[str, Any],
    portfolio_state: PortfolioStateProtocol
) -> bool:
    """Validate execution module configuration and dependencies.
    
    Args:
        execution_module: Execution module components
        portfolio_state: Expected portfolio state
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    required_components = [
        'execution_engine', 'broker', 'order_manager',
        'market_simulator', 'execution_context'
    ]
    
    # Check all required components exist
    for component in required_components:
        if component not in execution_module:
            raise ValueError(f"Missing required component: {component}")
    
    # Check broker portfolio state dependency
    broker = execution_module['broker']
    if hasattr(broker, '_portfolio_state'):
        if broker._portfolio_state is not portfolio_state:
            raise ValueError("Broker portfolio state mismatch")
    
    # Check component initialization
    for component_name, component in execution_module.items():
        if hasattr(component, '_initialized'):
            if not component._initialized:
                raise ValueError(f"Component not initialized: {component_name}")
    
    logger.info("Execution module validation passed")
    return True
