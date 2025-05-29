"""
Risk & Portfolio capabilities for container enhancement.
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal
import logging

from ..core.infrastructure.capabilities import Capability
from .risk_portfolio import RiskPortfolioContainer
from .position_sizing import (
    FixedPositionSizer,
    PercentagePositionSizer,
    VolatilityBasedSizer,
    KellyCriterionSizer,
    ATRBasedSizer
)
from .risk_limits import (
    MaxPositionLimit,
    MaxExposureLimit,
    MaxDrawdownLimit,
    VaRLimit,
    ConcentrationLimit,
    LeverageLimit,
    DailyLossLimit,
    SymbolRestrictionLimit
)

logger = logging.getLogger(__name__)


class RiskPortfolioCapability(Capability):
    """
    Adds unified Risk & Portfolio management functionality to containers.
    
    This capability transforms a container into a Risk & Portfolio container
    that can manage multiple strategy components and convert their signals
    into orders while enforcing risk limits.
    """
    
    def get_name(self) -> str:
        return "risk_portfolio"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """
        Apply Risk & Portfolio capability to component.
        
        Args:
            component: Component to enhance
            spec: Configuration including:
                - initial_capital: Starting capital
                - position_sizers: Position sizing configuration
                - risk_limits: Risk limit configuration
                - execution_mode: Execution context
                
        Returns:
            Enhanced component with Risk & Portfolio functionality
        """
        # Create Risk & Portfolio container
        initial_capital = Decimal(str(spec.get('initial_capital', 100000)))
        
        # Create risk portfolio instance
        risk_portfolio = RiskPortfolioContainer(
            container_id=f"{component.container_id}_risk_portfolio",
            parent_container=component,
            initial_capital=initial_capital
        )
        
        # Store as component attribute
        component.risk_portfolio = risk_portfolio
        
        # Configure position sizers
        self._configure_position_sizers(risk_portfolio, spec)
        
        # Configure risk limits
        self._configure_risk_limits(risk_portfolio, spec)
        
        # Add signal processing method
        def process_signal(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process signal through Risk & Portfolio."""
            return risk_portfolio.process_signal(signal)
        
        component.process_signal = process_signal
        
        # Add portfolio state access
        def get_portfolio_state() -> Dict[str, Any]:
            """Get current portfolio state."""
            return risk_portfolio.get_portfolio_state()
        
        component.get_portfolio_state = get_portfolio_state
        
        # Add position access
        def get_position(symbol: str) -> Optional[Any]:
            """Get position for symbol."""
            return risk_portfolio.portfolio_state.get_position(symbol)
        
        component.get_position = get_position
        
        # Add metrics access
        def get_risk_metrics() -> Dict[str, Any]:
            """Get current risk metrics."""
            return risk_portfolio.calculate_risk_metrics()
        
        component.get_risk_metrics = get_risk_metrics
        
        # Subscribe to parent events if event capability exists
        if hasattr(component, 'event_bus'):
            # Subscribe to SIGNAL events
            component.event_bus.subscribe(
                'SIGNAL',
                lambda event: risk_portfolio.process_signal(event.payload)
            )
            
            # Subscribe to FILL events
            component.event_bus.subscribe(
                'FILL',
                lambda event: risk_portfolio.handle_fill(event.payload)
            )
        
        # Add lifecycle hooks
        if hasattr(component, 'add_startup_hook'):
            component.add_startup_hook(risk_portfolio.initialize)
        
        if hasattr(component, 'add_shutdown_hook'):
            component.add_shutdown_hook(risk_portfolio.cleanup)
        
        logger.info(
            f"Applied Risk & Portfolio capability to {component.container_id}",
            extra={'initial_capital': float(initial_capital)}
        )
        
        return component
    
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
                {'name': 'default', 'type': 'fixed', 'size': 100}
            ]
        
        for config in sizer_configs:
            sizer = self._create_position_sizer(config)
            if sizer:
                risk_portfolio.add_position_sizer(
                    config['name'],
                    sizer
                )
    
    def _create_position_sizer(self, config: Dict[str, Any]) -> Any:
        """Create position sizer from configuration."""
        sizer_type = config.get('type')
        
        if sizer_type == 'fixed':
            return FixedPositionSizer(
                position_size=config.get('size', 100)
            )
        
        elif sizer_type == 'percentage':
            return PercentagePositionSizer(
                percentage=Decimal(str(config.get('percentage', 2.0)))
            )
        
        elif sizer_type == 'volatility':
            return VolatilityBasedSizer(
                risk_per_trade=Decimal(str(config.get('risk_per_trade', 1.0))),
                lookback_period=config.get('lookback_period', 20)
            )
        
        elif sizer_type == 'kelly':
            return KellyCriterionSizer(
                kelly_fraction=Decimal(str(config.get('kelly_fraction', 0.25))),
                max_leverage=Decimal(str(config.get('max_leverage', 1.0)))
            )
        
        elif sizer_type == 'atr':
            return ATRBasedSizer(
                risk_amount=Decimal(str(config.get('risk_amount', 1000))),
                atr_multiplier=Decimal(str(config.get('atr_multiplier', 2.0)))
            )
        
        else:
            logger.warning(f"Unknown position sizer type: {sizer_type}")
            return None
    
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
                {'type': 'position', 'max_position': 10000},
                {'type': 'exposure', 'max_exposure_pct': 20},
                {'type': 'drawdown', 'max_drawdown_pct': 10}
            ]
        
        for config in limit_configs:
            limit = self._create_risk_limit(config)
            if limit:
                risk_portfolio.add_risk_limit(limit)
    
    def _create_risk_limit(self, config: Dict[str, Any]) -> Any:
        """Create risk limit from configuration."""
        limit_type = config.get('type')
        
        if limit_type == 'position':
            return MaxPositionLimit(
                max_position=config.get('max_position', 10000)
            )
        
        elif limit_type == 'exposure':
            return MaxExposureLimit(
                max_exposure_pct=Decimal(str(config.get('max_exposure_pct', 20)))
            )
        
        elif limit_type == 'drawdown':
            return MaxDrawdownLimit(
                max_drawdown_pct=Decimal(str(config.get('max_drawdown_pct', 20))),
                reduce_at_pct=Decimal(str(config.get('reduce_at_pct', 15)))
            )
        
        elif limit_type == 'var':
            return VaRLimit(
                confidence_level=Decimal(str(config.get('confidence_level', 0.95))),
                max_var_pct=Decimal(str(config.get('max_var_pct', 5)))
            )
        
        elif limit_type == 'concentration':
            return ConcentrationLimit(
                max_position_pct=Decimal(str(config.get('max_position_pct', 10))),
                max_sector_pct=Decimal(str(config.get('max_sector_pct', 30)))
            )
        
        elif limit_type == 'leverage':
            return LeverageLimit(
                max_leverage=Decimal(str(config.get('max_leverage', 1.0)))
            )
        
        elif limit_type == 'daily_loss':
            return DailyLossLimit(
                max_daily_loss=Decimal(str(config.get('max_daily_loss', 5000))),
                max_daily_loss_pct=Decimal(str(config.get('max_daily_loss_pct', 5)))
            )
        
        elif limit_type == 'symbol_restriction':
            return SymbolRestrictionLimit(
                allowed_symbols=set(config.get('allowed_symbols', [])),
                blocked_symbols=set(config.get('blocked_symbols', []))
            )
        
        else:
            logger.warning(f"Unknown risk limit type: {limit_type}")
            return None


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
        
        # Check if thread safety needed
        if hasattr(component, 'requires_thread_safety') and component.requires_thread_safety():
            # Risk & Portfolio container already handles thread safety internally
            # based on its ExecutionContext, so no additional work needed
            logger.info(f"Thread-safe Risk & Portfolio enabled for {component.container_id}")
        
        return component