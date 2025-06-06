"""Reusable commission models for simulated brokers.

This module contains various commission calculation models that can be composed
with different broker implementations following Protocol + Composition.
"""

from typing import List, Tuple, Protocol
from decimal import Decimal

from ..protocols import Order
from ...core.containers.discovery import execution_model


class CommissionModel(Protocol):
    """Protocol for commission calculation models."""
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate commission for order."""
        ...


@execution_model(
    model_type='commission',
    name='zero',
    description='Zero commission for brokers like Alpaca'
)
class ZeroCommissionModel:
    """Zero commission model for brokers like Alpaca."""
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Always returns zero commission."""
        return Decimal("0.0")


@execution_model(
    model_type='commission',
    name='per_share',
    params={
        'rate_per_share': 0.005,
        'minimum_commission': 1.0,
        'maximum_commission': 10.0
    }
)
class PerShareCommissionModel:
    """Per-share commission model."""
    
    def __init__(
        self,
        rate_per_share: Decimal = Decimal("0.005"),
        minimum_commission: Decimal = Decimal("1.0"),
        maximum_commission: Decimal = Decimal("10.0")
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
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate per-share commission."""
        commission = fill_quantity * self.rate_per_share
        commission = max(commission, self.minimum_commission)
        commission = min(commission, self.maximum_commission)
        return commission


@execution_model(
    model_type='commission',
    name='percentage',
    params={
        'commission_percent': 0.001,
        'minimum_commission': 1.0
    }
)
class PercentageCommissionModel:
    """Percentage-based commission model."""
    
    def __init__(
        self,
        commission_percent: Decimal = Decimal("0.001"),  # 0.1%
        minimum_commission: Decimal = Decimal("1.0")
    ):
        """Initialize percentage commission model.
        
        Args:
            commission_percent: Commission as percentage of trade value
            minimum_commission: Minimum commission per trade
        """
        self.commission_percent = commission_percent
        self.minimum_commission = minimum_commission
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate percentage-based commission."""
        trade_value = fill_price * fill_quantity
        commission = trade_value * self.commission_percent
        return max(commission, self.minimum_commission)


@execution_model(
    model_type='commission',
    name='tiered',
    params={
        'minimum_commission': 1.0
    },
    description='Commission with tiered rates based on trade value'
)
class TieredCommissionModel:
    """Commission model with tiered rates based on trade value."""
    
    def __init__(
        self,
        tiers: List[Tuple[Decimal, Decimal]],
        minimum_commission: Decimal = Decimal("1.0")
    ):
        """Initialize tiered commission model.
        
        Args:
            tiers: List of (volume_threshold, rate) tuples, sorted by threshold
            minimum_commission: Minimum commission per trade
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.minimum_commission = minimum_commission
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate commission based on trade value tier."""
        trade_value = fill_price * fill_quantity
        
        # Find applicable tier
        commission_rate = self.tiers[0][1]  # Default to first tier
        for threshold, rate in self.tiers:
            if trade_value >= threshold:
                commission_rate = rate
            else:
                break
        
        commission = trade_value * commission_rate
        return max(commission, self.minimum_commission)


@execution_model(
    model_type='commission',
    name='fixed',
    params={'commission_per_trade': 1.0}
)
class FixedCommissionModel:
    """Fixed commission per trade model."""
    
    def __init__(self, commission_per_trade: Decimal = Decimal("1.0")):
        """Initialize fixed commission model.
        
        Args:
            commission_per_trade: Fixed commission amount per trade
        """
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate fixed commission."""
        return self.commission_per_trade


# Pre-configured commission models for common brokers
def create_alpaca_commission() -> CommissionModel:
    """Create Alpaca-style zero commission model."""
    return ZeroCommissionModel()


def create_interactive_brokers_commission() -> CommissionModel:
    """Create Interactive Brokers-style tiered commission model."""
    return TieredCommissionModel(
        tiers=[
            (Decimal("0"), Decimal("0.005")),      # $0+: 0.5%
            (Decimal("10000"), Decimal("0.003")),  # $10k+: 0.3%
            (Decimal("100000"), Decimal("0.001"))  # $100k+: 0.1%
        ],
        minimum_commission=Decimal("1.0")
    )


def create_traditional_broker_commission() -> CommissionModel:
    """Create traditional broker per-share commission model."""
    return PerShareCommissionModel(
        rate_per_share=Decimal("0.01"),  # $0.01 per share
        minimum_commission=Decimal("4.95"),
        maximum_commission=Decimal("29.95")
    )