"""Commission models for backtest execution.

Synchronous commission calculation models for market simulation.
"""

from typing import List, Tuple
from decimal import Decimal

from ...types import Order
from ...sync_protocols import CommissionModel


class ZeroCommissionModel:
    """Zero commission model for brokers like Alpaca."""
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Always returns zero commission."""
        return 0.0


class PerShareCommissionModel:
    """Per-share commission model."""
    
    def __init__(
        self,
        rate_per_share: float = 0.005,
        minimum_commission: float = 1.0,
        maximum_commission: float = 10.0
    ):
        """Initialize per-share commission model.
        
        Args:
            rate_per_share: Commission per share
            minimum_commission: Minimum commission per trade
            maximum_commission: Maximum commission per trade
        """
        self.rate_per_share = rate_per_share
        self.minimum_commission = minimum_commission
        self.maximum_commission = maximum_commission
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate per-share commission."""
        commission = float(order.quantity) * self.rate_per_share
        commission = max(commission, self.minimum_commission)
        commission = min(commission, self.maximum_commission)
        return commission


class PercentageCommissionModel:
    """Percentage-based commission model."""
    
    def __init__(
        self,
        commission_percent: float = 0.001,  # 0.1%
        minimum_commission: float = 1.0
    ):
        """Initialize percentage commission model.
        
        Args:
            commission_percent: Commission as percentage of trade value
            minimum_commission: Minimum commission per trade
        """
        self.commission_percent = commission_percent
        self.minimum_commission = minimum_commission
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate percentage-based commission."""
        trade_value = fill_price * float(order.quantity)
        commission = trade_value * self.commission_percent
        return max(commission, self.minimum_commission)


class TieredCommissionModel:
    """Commission model with tiered rates based on trade value."""
    
    def __init__(
        self,
        tiers: List[Tuple[float, float]] = None,
        minimum_commission: float = 1.0
    ):
        """Initialize tiered commission model.
        
        Args:
            tiers: List of (volume_threshold, rate) tuples, sorted by threshold
            minimum_commission: Minimum commission per trade
        """
        if tiers is None:
            # Default Interactive Brokers-style tiers
            tiers = [
                (0.0, 0.005),      # $0+: 0.5%
                (10000.0, 0.003),  # $10k+: 0.3%
                (100000.0, 0.001)  # $100k+: 0.1%
            ]
        
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.minimum_commission = minimum_commission
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission based on trade value tier."""
        trade_value = fill_price * float(order.quantity)
        
        # Find applicable tier
        commission_rate = self.tiers[0][1]  # Default to first tier
        for threshold, rate in self.tiers:
            if trade_value >= threshold:
                commission_rate = rate
            else:
                break
        
        commission = trade_value * commission_rate
        return max(commission, self.minimum_commission)


class FixedCommissionModel:
    """Fixed commission per trade model."""
    
    def __init__(self, commission_per_trade: float = 1.0):
        """Initialize fixed commission model.
        
        Args:
            commission_per_trade: Fixed commission amount per trade
        """
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate fixed commission."""
        return self.commission_per_trade