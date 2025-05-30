"""Risk limit implementations for portfolio protection."""

from abc import ABC
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta

from .protocols import (
    RiskLimitProtocol,
    Order,
    PortfolioStateProtocol,
    Position,
)


class BaseRiskLimit(ABC, RiskLimitProtocol):
    """Base class for risk limits."""
    
    def __init__(self, name: str):
        """Initialize base risk limit.
        
        Args:
            name: Name of the risk limit
        """
        self.name = name
        self._violations: List[Dict[str, Any]] = []
    
    def _record_violation(self, order: Order, reason: str) -> None:
        """Record a limit violation.
        
        Args:
            order: Order that violated the limit
            reason: Reason for violation
        """
        self._violations.append({
            "timestamp": datetime.now(),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "reason": reason
        })
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get recent violations."""
        # Return last 100 violations
        return self._violations[-100:]


class MaxPositionLimit(BaseRiskLimit):
    """Maximum position size limit."""
    
    def __init__(
        self,
        max_position_value: Optional[Decimal] = None,
        max_position_percent: Optional[Decimal] = None,
        name: str = "MaxPositionLimit"
    ):
        """Initialize maximum position limit.
        
        Args:
            max_position_value: Maximum position value in base currency
            max_position_percent: Maximum position as % of portfolio
            name: Name of the limit
        """
        super().__init__(name)
        self.max_position_value = max_position_value
        self.max_position_percent = max_position_percent
        
        if not (max_position_value or max_position_percent):
            raise ValueError("Must specify either max_position_value or max_position_percent")
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if order violates position size limit.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        # Get current position
        current_position = portfolio_state.get_position(order.symbol)
        current_quantity = current_position.quantity if current_position else Decimal(0)
        
        # Calculate new position
        new_quantity = current_quantity
        if order.side.value == "buy":
            new_quantity += order.quantity
        else:
            new_quantity -= order.quantity
        
        # Get price
        price = order.price
        if not price:  # Market order
            price = market_data.get("prices", {}).get(order.symbol)
            if not price:
                return False, "No price available for position limit check"
        
        price = Decimal(str(price))
        new_position_value = abs(new_quantity) * price
        
        # Check value limit
        if self.max_position_value and new_position_value > self.max_position_value:
            reason = f"Position value {new_position_value} exceeds limit {self.max_position_value}"
            self._record_violation(order, reason)
            return False, reason
        
        # Check percentage limit
        if self.max_position_percent:
            portfolio_value = portfolio_state.get_total_value()
            if portfolio_value == 0:
                return False, "Cannot calculate position percentage with zero portfolio value"
            position_percent = new_position_value / portfolio_value
            
            if position_percent > self.max_position_percent:
                reason = f"Position {position_percent:.1%} exceeds limit {self.max_position_percent:.1%}"
                self._record_violation(order, reason)
                return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_position_value": str(self.max_position_value) if self.max_position_value else None,
            "max_position_percent": str(self.max_position_percent) if self.max_position_percent else None,
            "violations_count": len(self._violations)
        }


class MaxDrawdownLimit(BaseRiskLimit):
    """Maximum drawdown limit."""
    
    def __init__(
        self,
        max_drawdown: Decimal,
        lookback_days: Optional[int] = None,
        name: str = "MaxDrawdownLimit"
    ):
        """Initialize maximum drawdown limit.
        
        Args:
            max_drawdown: Maximum allowed drawdown (0-1)
            lookback_days: Lookback period for drawdown calculation
            name: Name of the limit
        """
        super().__init__(name)
        self.max_drawdown = max_drawdown
        self.lookback_days = lookback_days
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if current drawdown exceeds limit.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        metrics = portfolio_state.get_risk_metrics()
        current_drawdown = metrics.current_drawdown
        
        if current_drawdown > self.max_drawdown:
            reason = f"Current drawdown {current_drawdown:.1%} exceeds limit {self.max_drawdown:.1%}"
            self._record_violation(order, reason)
            return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_drawdown": str(self.max_drawdown),
            "lookback_days": self.lookback_days,
            "violations_count": len(self._violations)
        }


class VaRLimit(BaseRiskLimit):
    """Value at Risk (VaR) limit."""
    
    def __init__(
        self,
        max_var: Decimal,
        confidence_level: Decimal = Decimal("0.95"),
        name: str = "VaRLimit"
    ):
        """Initialize VaR limit.
        
        Args:
            max_var: Maximum VaR as fraction of portfolio
            confidence_level: Confidence level for VaR (e.g., 0.95)
            name: Name of the limit
        """
        super().__init__(name)
        self.max_var = max_var
        self.confidence_level = confidence_level
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if VaR exceeds limit.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        metrics = portfolio_state.get_risk_metrics()
        
        # Check if VaR is available
        if metrics.var_95 is None:
            # Can't check, allow order
            return True, None
        
        # Simple check: current VaR
        portfolio_value = portfolio_state.get_total_value()
        var_fraction = metrics.var_95 / portfolio_value
        
        if var_fraction > self.max_var:
            reason = f"VaR {var_fraction:.1%} exceeds limit {self.max_var:.1%}"
            self._record_violation(order, reason)
            return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_var": str(self.max_var),
            "confidence_level": str(self.confidence_level),
            "violations_count": len(self._violations)
        }


class MaxExposureLimit(BaseRiskLimit):
    """Maximum total exposure limit."""
    
    def __init__(
        self,
        max_exposure_pct: Decimal,
        name: str = "MaxExposureLimit"
    ):
        """Initialize maximum exposure limit.
        
        Args:
            max_exposure_pct: Maximum exposure as % of portfolio (0-100)
            name: Name of the limit
        """
        super().__init__(name)
        self.max_exposure_pct = max_exposure_pct
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if order would exceed total exposure limit.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        # Get current metrics
        metrics = portfolio_state.get_risk_metrics()
        portfolio_value = portfolio_state.get_total_value()
        current_exposure = metrics.positions_value
        
        # Calculate order value
        price = order.price
        if not price:
            price = market_data.get("prices", {}).get(order.symbol)
            if not price:
                return False, "No price available for exposure check"
        
        price = Decimal(str(price))
        order_value = order.quantity * price
        
        # Calculate new exposure
        if order.side.value == "buy":
            new_exposure = current_exposure + order_value
        else:
            # Selling reduces exposure
            new_exposure = current_exposure - order_value
        
        # Calculate exposure percentage
        exposure_pct = (new_exposure / portfolio_value) * 100 if portfolio_value > 0 else Decimal(0)
        
        if exposure_pct > self.max_exposure_pct:
            reason = f"Total exposure {exposure_pct:.1f}% would exceed limit {self.max_exposure_pct:.1f}%"
            self._record_violation(order, reason)
            return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_exposure_pct": str(self.max_exposure_pct),
            "violations_count": len(self._violations)
        }


class ConcentrationLimit(BaseRiskLimit):
    """Portfolio concentration limit."""
    
    def __init__(
        self,
        max_single_position: Decimal = Decimal("0.25"),
        max_sector_exposure: Optional[Decimal] = None,
        max_correlated_exposure: Optional[Decimal] = None,
        name: str = "ConcentrationLimit"
    ):
        """Initialize concentration limit.
        
        Args:
            max_single_position: Max % in single position
            max_sector_exposure: Max % in single sector
            max_correlated_exposure: Max % in correlated assets
            name: Name of the limit
        """
        super().__init__(name)
        self.max_single_position = max_single_position
        self.max_sector_exposure = max_sector_exposure
        self.max_correlated_exposure = max_correlated_exposure
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if order violates concentration limits.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        # Get portfolio metrics
        metrics = portfolio_state.get_risk_metrics()
        portfolio_value = portfolio_state.get_total_value()
        
        # Calculate position after order
        current_position = portfolio_state.get_position(order.symbol)
        current_value = current_position.market_value if current_position else Decimal(0)
        
        # Get order value
        price = order.price
        if not price:
            price = market_data.get("prices", {}).get(order.symbol)
            if not price:
                return False, "No price available for concentration check"
        
        price = Decimal(str(price))
        order_value = order.quantity * price
        
        if order.side.value == "buy":
            new_value = current_value + order_value
        else:
            new_value = current_value - order_value
        
        if portfolio_value == 0:
            return False, "Cannot calculate concentration with zero portfolio value"
        new_concentration = abs(new_value) / portfolio_value
        
        # Check single position limit
        if new_concentration > self.max_single_position:
            reason = f"Position concentration {new_concentration:.1%} exceeds limit {self.max_single_position:.1%}"
            self._record_violation(order, reason)
            return False, reason
        
        # TODO: Add sector and correlation checks when market data includes this info
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_single_position": str(self.max_single_position),
            "max_sector_exposure": str(self.max_sector_exposure) if self.max_sector_exposure else None,
            "max_correlated_exposure": str(self.max_correlated_exposure) if self.max_correlated_exposure else None,
            "violations_count": len(self._violations)
        }


class LeverageLimit(BaseRiskLimit):
    """Portfolio leverage limit."""
    
    def __init__(
        self,
        max_leverage: Decimal = Decimal("1.0"),
        name: str = "LeverageLimit"
    ):
        """Initialize leverage limit.
        
        Args:
            max_leverage: Maximum allowed leverage
            name: Name of the limit
        """
        super().__init__(name)
        self.max_leverage = max_leverage
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if order would exceed leverage limit.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        # Get current leverage
        metrics = portfolio_state.get_risk_metrics()
        current_leverage = metrics.leverage
        
        # For buy orders, check if we'd exceed leverage
        if order.side.value == "buy":
            # Get order value
            price = order.price
            if not price:
                price = market_data.get("prices", {}).get(order.symbol)
                if not price:
                    return False, "No price available for leverage check"
            
            price = Decimal(str(price))
            order_value = order.quantity * price
            
            # Calculate new positions value
            new_positions_value = metrics.positions_value + order_value
            
            # Calculate new leverage
            cash = portfolio_state.get_cash_balance()
            equity = metrics.total_value
            
            # Simple leverage calculation
            if cash < order_value:
                # Would need margin
                margin_needed = order_value - cash
                new_leverage = new_positions_value / equity
                
                if new_leverage > self.max_leverage:
                    reason = f"Leverage {new_leverage:.2f}x exceeds limit {self.max_leverage:.2f}x"
                    self._record_violation(order, reason)
                    return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_leverage": str(self.max_leverage),
            "violations_count": len(self._violations)
        }


class DailyLossLimit(BaseRiskLimit):
    """Daily loss limit."""
    
    def __init__(
        self,
        max_daily_loss: Decimal,
        name: str = "DailyLossLimit"
    ):
        """Initialize daily loss limit.
        
        Args:
            max_daily_loss: Maximum daily loss as fraction of portfolio
            name: Name of the limit
        """
        super().__init__(name)
        self.max_daily_loss = max_daily_loss
        self._daily_pnl: Dict[str, Decimal] = {}
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if daily loss limit is exceeded.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        # Get today's date
        today = datetime.now().date().isoformat()
        
        # Get current daily P&L (simplified - would need proper tracking)
        metrics = portfolio_state.get_risk_metrics()
        portfolio_value = portfolio_state.get_total_value()
        
        # Check if we have daily P&L tracking
        daily_pnl = self._daily_pnl.get(today, Decimal(0))
        
        # Simple check: if unrealized P&L is very negative
        total_pnl = metrics.unrealized_pnl + metrics.realized_pnl
        daily_loss_fraction = -total_pnl / portfolio_value if total_pnl < 0 else Decimal(0)
        
        if daily_loss_fraction > self.max_daily_loss:
            reason = f"Daily loss {daily_loss_fraction:.1%} exceeds limit {self.max_daily_loss:.1%}"
            self._record_violation(order, reason)
            return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "max_daily_loss": str(self.max_daily_loss),
            "violations_count": len(self._violations)
        }


class SymbolRestrictionLimit(BaseRiskLimit):
    """Restrict trading to specific symbols."""
    
    def __init__(
        self,
        allowed_symbols: Optional[Set[str]] = None,
        blocked_symbols: Optional[Set[str]] = None,
        name: str = "SymbolRestrictionLimit"
    ):
        """Initialize symbol restriction limit.
        
        Args:
            allowed_symbols: Set of allowed symbols (if None, all allowed)
            blocked_symbols: Set of blocked symbols
            name: Name of the limit
        """
        super().__init__(name)
        self.allowed_symbols = allowed_symbols
        self.blocked_symbols = blocked_symbols or set()
    
    def check_limit(
        self,
        order: Order,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if symbol is allowed for trading.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        # Check blocked symbols
        if order.symbol in self.blocked_symbols:
            reason = f"Symbol {order.symbol} is blocked for trading"
            self._record_violation(order, reason)
            return False, reason
        
        # Check allowed symbols
        if self.allowed_symbols and order.symbol not in self.allowed_symbols:
            reason = f"Symbol {order.symbol} is not in allowed list"
            self._record_violation(order, reason)
            return False, reason
        
        return True, None
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        return {
            "name": self.name,
            "allowed_symbols_count": len(self.allowed_symbols) if self.allowed_symbols else "all",
            "blocked_symbols": list(self.blocked_symbols),
            "violations_count": len(self._violations)
        }