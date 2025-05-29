"""Position sizing strategies for risk management."""

from abc import ABC
from decimal import Decimal
from typing import Dict, Any, Optional
import math

from .protocols import (
    PositionSizerProtocol,
    Signal,
    PortfolioStateProtocol,
)


class BasePositionSizer(ABC, PositionSizerProtocol):
    """Base class for position sizing strategies."""
    
    def __init__(self, min_size: Decimal = Decimal("0.01")):
        """Initialize base position sizer.
        
        Args:
            min_size: Minimum position size
        """
        self.min_size = min_size
    
    def _apply_constraints(self, size: Decimal) -> Decimal:
        """Apply basic constraints to position size.
        
        Args:
            size: Raw position size
            
        Returns:
            Constrained position size
        """
        # Ensure minimum size
        if abs(size) < self.min_size:
            return Decimal(0)
        
        # Round to reasonable precision (e.g., shares)
        return size.quantize(Decimal("0.01"))


class FixedPositionSizer(BasePositionSizer):
    """Fixed position size strategy."""
    
    def __init__(self, size: Decimal, min_size: Decimal = Decimal("0.01")):
        """Initialize fixed position sizer.
        
        Args:
            size: Fixed position size
            min_size: Minimum position size
        """
        super().__init__(min_size)
        self.size = size
    
    def calculate_size(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate fixed position size.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Fixed position size
        """
        # Adjust size based on signal strength
        adjusted_size = self.size * abs(signal.strength)
        return self._apply_constraints(adjusted_size)


class PercentagePositionSizer(BasePositionSizer):
    """Percentage of portfolio position sizing."""
    
    def __init__(
        self,
        percentage: Decimal,
        use_leverage: bool = False,
        min_size: Decimal = Decimal("0.01")
    ):
        """Initialize percentage position sizer.
        
        Args:
            percentage: Percentage of portfolio per position (0-1)
            use_leverage: Whether to allow leveraged positions
            min_size: Minimum position size
        """
        super().__init__(min_size)
        self.percentage = percentage
        self.use_leverage = use_leverage
    
    def calculate_size(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate position size as percentage of portfolio.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Position size based on portfolio percentage
        """
        # Get current portfolio value
        portfolio_value = portfolio_state.get_total_value()
        
        # Get current price
        price = market_data.get("prices", {}).get(signal.symbol)
        if not price:
            return Decimal(0)
        
        price = Decimal(str(price))
        
        # Calculate base position value
        position_value = portfolio_value * self.percentage
        
        # Adjust for signal strength
        position_value *= abs(signal.strength)
        
        # Check leverage constraints
        if not self.use_leverage:
            cash_available = portfolio_state.get_cash_balance()
            position_value = min(position_value, cash_available)
        
        # Convert to shares
        shares = position_value / price
        
        return self._apply_constraints(shares)


class KellyCriterionSizer(BasePositionSizer):
    """Kelly Criterion based position sizing."""
    
    def __init__(
        self,
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        kelly_fraction: Decimal = Decimal("0.25"),
        max_percentage: Decimal = Decimal("0.25"),
        min_size: Decimal = Decimal("0.01")
    ):
        """Initialize Kelly Criterion sizer.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            kelly_fraction: Fraction of Kelly to use (safety)
            max_percentage: Maximum percentage of portfolio
            min_size: Minimum position size
        """
        super().__init__(min_size)
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
        self.max_percentage = max_percentage
    
    def calculate_size(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate position size using Kelly Criterion.
        
        Kelly formula: f = (p*b - q) / b
        where:
        - f = fraction of capital to bet
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = ratio of win to loss
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Kelly-optimized position size
        """
        # Calculate Kelly percentage
        b = self.avg_win / self.avg_loss
        q = Decimal(1) - self.win_rate
        
        kelly_percentage = (self.win_rate * b - q) / b
        
        # Apply Kelly fraction for safety
        kelly_percentage *= self.kelly_fraction
        
        # Apply maximum constraint
        kelly_percentage = min(kelly_percentage, self.max_percentage)
        
        # Adjust for signal strength
        kelly_percentage *= abs(signal.strength)
        
        # Get portfolio value and price
        portfolio_value = portfolio_state.get_total_value()
        price = market_data.get("prices", {}).get(signal.symbol)
        if not price:
            return Decimal(0)
        
        price = Decimal(str(price))
        
        # Calculate position value and shares
        position_value = portfolio_value * kelly_percentage
        shares = position_value / price
        
        return self._apply_constraints(shares)


class VolatilityBasedSizer(BasePositionSizer):
    """Volatility-based position sizing (risk parity approach)."""
    
    def __init__(
        self,
        target_volatility: Decimal,
        lookback_days: int = 20,
        max_percentage: Decimal = Decimal("0.25"),
        min_size: Decimal = Decimal("0.01")
    ):
        """Initialize volatility-based sizer.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            lookback_days: Days for volatility calculation
            max_percentage: Maximum percentage of portfolio
            min_size: Minimum position size
        """
        super().__init__(min_size)
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.max_percentage = max_percentage
    
    def calculate_size(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate position size based on volatility targeting.
        
        Size inversely proportional to asset volatility to achieve
        equal risk contribution.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Volatility-adjusted position size
        """
        # Get asset volatility from market data
        volatility_data = market_data.get("volatility", {})
        asset_volatility = volatility_data.get(signal.symbol)
        
        if not asset_volatility:
            # Fallback to simple percentage sizing
            return PercentagePositionSizer(
                percentage=Decimal("0.02")
            ).calculate_size(signal, portfolio_state, market_data)
        
        asset_volatility = Decimal(str(asset_volatility))
        
        # Calculate position percentage based on volatility
        # Lower volatility = larger position
        position_percentage = self.target_volatility / asset_volatility
        
        # Apply maximum constraint
        position_percentage = min(position_percentage, self.max_percentage)
        
        # Adjust for signal strength
        position_percentage *= abs(signal.strength)
        
        # Get portfolio value and price
        portfolio_value = portfolio_state.get_total_value()
        price = market_data.get("prices", {}).get(signal.symbol)
        if not price:
            return Decimal(0)
        
        price = Decimal(str(price))
        
        # Calculate position value and shares
        position_value = portfolio_value * position_percentage
        shares = position_value / price
        
        return self._apply_constraints(shares)


class ATRBasedSizer(BasePositionSizer):
    """ATR (Average True Range) based position sizing."""
    
    def __init__(
        self,
        risk_per_trade: Decimal,
        atr_multiplier: Decimal = Decimal("2"),
        max_percentage: Decimal = Decimal("0.25"),
        min_size: Decimal = Decimal("0.01")
    ):
        """Initialize ATR-based sizer.
        
        Args:
            risk_per_trade: Risk per trade as fraction of portfolio
            atr_multiplier: ATR multiplier for stop loss
            max_percentage: Maximum percentage of portfolio
            min_size: Minimum position size
        """
        super().__init__(min_size)
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_percentage = max_percentage
    
    def calculate_size(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate position size based on ATR and fixed risk.
        
        Position size = Risk Amount / (ATR * Multiplier)
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            ATR-based position size
        """
        # Get ATR from market data
        atr_data = market_data.get("atr", {})
        atr = atr_data.get(signal.symbol)
        
        if not atr:
            # Fallback to percentage sizing
            return PercentagePositionSizer(
                percentage=self.max_percentage * Decimal("0.1")
            ).calculate_size(signal, portfolio_state, market_data)
        
        atr = Decimal(str(atr))
        
        # Get portfolio value and price
        portfolio_value = portfolio_state.get_total_value()
        price = market_data.get("prices", {}).get(signal.symbol)
        if not price:
            return Decimal(0)
        
        price = Decimal(str(price))
        
        # Calculate risk amount
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Calculate stop distance
        stop_distance = atr * self.atr_multiplier
        
        # Calculate shares based on risk
        shares = risk_amount / stop_distance
        
        # Check maximum position constraint
        position_value = shares * price
        max_value = portfolio_value * self.max_percentage
        
        if position_value > max_value:
            shares = max_value / price
        
        # Adjust for signal strength
        shares *= abs(signal.strength)
        
        return self._apply_constraints(shares)