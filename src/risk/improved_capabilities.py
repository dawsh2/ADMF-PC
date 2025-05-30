"""
Improved Risk & Portfolio capabilities with proper integration.

This module provides capabilities that integrate seamlessly with the core
capability system and dependency injection container.
"""

from typing import Any, Dict, List, Optional, Type, Set
from decimal import Decimal
import logging

from ..core.components.protocols import Capability
from ..core.dependencies.container import DependencyContainer
from .improved_risk_portfolio import (
    RiskPortfolioContainer,
    create_risk_portfolio_container
)
from .dependency_injection import (
    RiskDependencyResolver,
    RiskComponentFactory,
    create_position_sizer_spec,
    create_risk_limit_spec,
    create_signal_processor_spec,
)

logger = logging.getLogger(__name__)


class RiskPortfolioCapability(Capability):
    """
    Enhanced Risk & Portfolio capability with proper DI integration.
    
    This capability transforms a container into a Risk & Portfolio container
    with full dependency injection support and component lifecycle management.
    """
    
    def get_name(self) -> str:
        return "risk_portfolio"
    
    def get_required_capabilities(self) -> Set[Type[Capability]]:
        """Get required capabilities."""
        return set()  # No dependencies on other capabilities
    
    def get_provided_capabilities(self) -> Set[Type[Capability]]:
        """Get provided capabilities."""
        return {RiskPortfolioCapability}
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """
        Apply Risk & Portfolio capability with proper DI.
        
        Args:
            component: Component to enhance
            spec: Configuration including:
                - initial_capital: Starting capital
                - position_sizers: Position sizing configuration
                - risk_limits: Risk limit configuration
                - signal_processor: Signal processor configuration
                
        Returns:
            Enhanced component with Risk & Portfolio functionality
        """
        # Get or create dependency container
        dependency_container = self._get_dependency_container(component, spec)
        
        # Create Risk & Portfolio container
        initial_capital = Decimal(str(spec.get('initial_capital', 100000)))
        component_id = getattr(component, 'component_id', 'risk_portfolio')
        
        risk_portfolio = create_risk_portfolio_container(
            component_id=f"{component_id}_risk_portfolio",
            dependency_container=dependency_container,
            initial_capital=initial_capital,
            base_currency=spec.get('base_currency', 'USD')
        )
        
        # Configure components
        self._configure_signal_processor(risk_portfolio, spec)
        self._configure_position_sizers(risk_portfolio, spec)
        self._configure_risk_limits(risk_portfolio, spec)
        
        # Attach to component
        component.risk_portfolio = risk_portfolio
        
        # Add delegation methods for easier access
        self._add_delegation_methods(component, risk_portfolio)
        
        # Setup lifecycle integration
        self._setup_lifecycle_integration(component, risk_portfolio)
        
        # Setup event integration
        self._setup_event_integration(component, risk_portfolio)
        
        logger.info(
            f"Applied Risk & Portfolio capability to {component_id} "
            f"with initial capital: {float(initial_capital)}"
        )
        
        return component
    
    def _get_dependency_container(
        self,
        component: Any,
        spec: Dict[str, Any]
    ) -> DependencyContainer:
        """Get or create dependency container."""
        # Check if component already has a dependency container
        if hasattr(component, '_dependency_container'):
            return component._dependency_container
        
        # Check if component has a container reference
        if hasattr(component, 'container') and hasattr(component.container, '_dependency_container'):
            return component.container._dependency_container
        
        # Create new container
        container_id = getattr(component, 'component_id', 'risk_portfolio')
        dependency_container = DependencyContainer(container_id=container_id)
        
        # Store reference
        component._dependency_container = dependency_container
        
        return dependency_container
    
    def _configure_signal_processor(
        self,
        risk_portfolio: RiskPortfolioContainer,
        spec: Dict[str, Any]
    ) -> None:
        """Configure signal processor."""
        processor_config = spec.get('signal_processor', {})
        processor_type = processor_config.get('type', 'standard')
        processor_params = processor_config.get('params', {})
        
        risk_portfolio.configure_signal_processor(
            processor_type=processor_type,
            **processor_params
        )
    
    def _configure_position_sizers(
        self,
        risk_portfolio: RiskPortfolioContainer,
        spec: Dict[str, Any]
    ) -> None:
        """Configure position sizers from spec."""
        sizer_configs = spec.get('position_sizers', [])
        
        # Add default if none specified
        if not sizer_configs:
            sizer_configs = [
                {'name': 'default', 'type': 'percentage', 'percentage': 2.0}
            ]
        
        for config in sizer_configs:
            name = config.pop('name')
            sizer_type = config.pop('type')
            
            # Convert numeric parameters to Decimal
            decimal_params = {}
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    decimal_params[key] = Decimal(str(value))
                else:
                    decimal_params[key] = value
            
            risk_portfolio.configure_position_sizer(
                name=name,
                sizer_type=sizer_type,
                **decimal_params
            )
    
    def _configure_risk_limits(
        self,
        risk_portfolio: RiskPortfolioContainer,
        spec: Dict[str, Any]
    ) -> None:
        """Configure risk limits from spec."""
        limit_configs = spec.get('risk_limits', [])
        
        # Add default limits if none specified
        if not limit_configs:
            limit_configs = [
                {'type': 'position', 'max_position_value': 10000},
                {'type': 'exposure', 'max_exposure_pct': 20},
                {'type': 'drawdown', 'max_drawdown_pct': 10}
            ]
        
        for config in limit_configs:
            limit_type = config.pop('type')
            name = config.pop('name', None)
            
            # Convert numeric parameters to Decimal
            decimal_params = {}
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    decimal_params[key] = Decimal(str(value))
                else:
                    decimal_params[key] = value
            
            risk_portfolio.configure_risk_limit(
                limit_type=limit_type,
                name=name,
                **decimal_params
            )
    
    def _add_delegation_methods(
        self,
        component: Any,
        risk_portfolio: RiskPortfolioContainer
    ) -> None:
        """Add delegation methods for easier access."""
        
        def process_signals(signals, market_data):
            """Process signals through Risk & Portfolio."""
            return risk_portfolio.process_signals(signals, market_data)
        
        def get_portfolio_state():
            """Get current portfolio state."""
            return risk_portfolio.get_portfolio_state()
        
        def get_position(symbol: str):
            """Get position for symbol."""
            return risk_portfolio.get_portfolio_state().get_position(symbol)
        
        def get_risk_metrics():
            """Get current risk metrics."""
            return risk_portfolio.get_portfolio_state().get_risk_metrics()
        
        def get_risk_report():
            """Get comprehensive risk report."""
            return risk_portfolio.get_risk_report()
        
        def update_fills(fills):
            """Update portfolio with fills."""
            return risk_portfolio.update_fills(fills)
        
        def update_market_data(market_data):
            """Update market data."""
            return risk_portfolio.update_market_data(market_data)
        
        # Attach methods
        component.process_signals = process_signals
        component.get_portfolio_state = get_portfolio_state
        component.get_position = get_position
        component.get_risk_metrics = get_risk_metrics
        component.get_risk_report = get_risk_report
        component.update_fills = update_fills
        component.update_market_data = update_market_data
    
    def _setup_lifecycle_integration(
        self,
        component: Any,
        risk_portfolio: RiskPortfolioContainer
    ) -> None:
        """Setup lifecycle integration."""
        # Store original lifecycle methods
        original_initialize = getattr(component, 'initialize', None)
        original_start = getattr(component, 'start', None)
        original_stop = getattr(component, 'stop', None)
        original_reset = getattr(component, 'reset', None)
        original_teardown = getattr(component, 'teardown', None)
        
        def enhanced_initialize(context):
            """Enhanced initialize method."""
            # Initialize risk portfolio first
            risk_portfolio.initialize(context)
            
            # Call original method
            if original_initialize:
                original_initialize(context)
        
        def enhanced_start():
            """Enhanced start method."""
            # Start risk portfolio first
            risk_portfolio.start()
            
            # Call original method
            if original_start:
                original_start()
        
        def enhanced_stop():
            """Enhanced stop method."""
            # Call original method first
            if original_stop:
                original_stop()
            
            # Stop risk portfolio
            risk_portfolio.stop()
        
        def enhanced_reset():
            """Enhanced reset method."""
            # Reset risk portfolio first
            risk_portfolio.reset()
            
            # Call original method
            if original_reset:
                original_reset()
        
        def enhanced_teardown():
            """Enhanced teardown method."""
            # Call original method first
            if original_teardown:
                original_teardown()
            
            # Teardown risk portfolio
            risk_portfolio.teardown()
        
        # Replace methods
        component.initialize = enhanced_initialize
        component.start = enhanced_start
        component.stop = enhanced_stop
        component.reset = enhanced_reset
        component.teardown = enhanced_teardown
    
    def _setup_event_integration(
        self,
        component: Any,
        risk_portfolio: RiskPortfolioContainer
    ) -> None:
        """Setup event system integration."""
        if not hasattr(component, 'event_bus'):
            return
        
        # Set event bus on risk portfolio
        risk_portfolio.event_bus = component.event_bus
        
        # Subscribe to relevant events
        try:
            # Subscribe to SIGNAL events
            component.event_bus.subscribe(
                'SIGNAL',
                lambda event: self._handle_signal_event(risk_portfolio, event)
            )
            
            # Subscribe to FILL events
            component.event_bus.subscribe(
                'FILL',
                lambda event: self._handle_fill_event(risk_portfolio, event)
            )
            
            # Subscribe to MARKET_DATA events
            component.event_bus.subscribe(
                'MARKET_DATA',
                lambda event: self._handle_market_data_event(risk_portfolio, event)
            )
            
        except AttributeError:
            # Event bus might not support subscription
            logger.warning("Event bus does not support subscriptions")
    
    def _handle_signal_event(
        self,
        risk_portfolio: RiskPortfolioContainer,
        event: Any
    ) -> None:
        """Handle SIGNAL events."""
        try:
            signal = event.payload.get('signal')
            market_data = event.payload.get('market_data', {})
            
            if signal:
                # Process single signal
                orders = risk_portfolio.process_signals([signal], market_data)
                
                # Emit ORDER events for each order
                for order in orders:
                    if risk_portfolio.event_bus:
                        order_event = {
                            'event_type': 'ORDER',
                            'source_id': risk_portfolio.component_id,
                            'payload': {'order': order}
                        }
                        # Would publish order event here
        except Exception as e:
            logger.error(f"Error handling signal event: {e}")
    
    def _handle_fill_event(
        self,
        risk_portfolio: RiskPortfolioContainer,
        event: Any
    ) -> None:
        """Handle FILL events."""
        try:
            fill = event.payload.get('fill')
            if fill:
                risk_portfolio.update_fills([fill])
        except Exception as e:
            logger.error(f"Error handling fill event: {e}")
    
    def _handle_market_data_event(
        self,
        risk_portfolio: RiskPortfolioContainer,
        event: Any
    ) -> None:
        """Handle MARKET_DATA events."""
        try:
            market_data = event.payload
            if market_data:
                risk_portfolio.update_market_data(market_data)
        except Exception as e:
            logger.error(f"Error handling market data event: {e}")


class PositionSizingCapability(Capability):
    """Capability for position sizing strategies."""
    
    def get_name(self) -> str:
        return "position_sizing"


class RiskLimitCapability(Capability):
    """Capability for risk limit enforcement."""
    
    def get_name(self) -> str:
        return "risk_limits"


class PortfolioTrackingCapability(Capability):
    """Capability for portfolio state tracking."""
    
    def get_name(self) -> str:
        return "portfolio_tracking"


class ThreadSafeRiskPortfolioCapability(RiskPortfolioCapability):
    """
    Thread-safe version of Risk & Portfolio capability.
    
    Automatically applied when ExecutionContext requires thread safety.
    """
    
    def get_name(self) -> str:
        return "thread_safe_risk_portfolio"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply thread-safe Risk & Portfolio capability."""
        # First apply base capability
        component = super().apply(component, spec)
        
        # Thread safety is handled internally by the risk portfolio container
        # based on signal volume and processing patterns
        
        logger.info(f"Thread-safe Risk & Portfolio enabled for {component.component_id}")
        
        return component


# Export for capability registration
__all__ = [
    'RiskPortfolioCapability',
    'PositionSizingCapability', 
    'RiskLimitCapability',
    'PortfolioTrackingCapability',
    'ThreadSafeRiskPortfolioCapability'
]
