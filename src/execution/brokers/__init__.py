"""
Simulated broker implementations with composable models.

This module provides the canonical SimulatedBroker implementation following 
Protocol + Composition principles, along with factory functions for common scenarios.
"""

from .simulated import (
    SimulatedBroker,
    PortfolioStateProtocol,
    MarketDataProtocol,
    create_simulated_broker,
    create_zero_commission_broker,
    create_traditional_broker,
    create_conservative_broker,
)

from .slippage import (
    SlippageModel,
    MarketConditions,
    PercentageSlippageModel,
    VolumeImpactSlippageModel,
    FixedSlippageModel,
    ZeroSlippageModel,
)

from .commission import (
    CommissionModel,
    ZeroCommissionModel,
    PerShareCommissionModel,
    PercentageCommissionModel,
    TieredCommissionModel,
    FixedCommissionModel,
    create_alpaca_commission,
    create_interactive_brokers_commission,
    create_traditional_broker_commission,
)

from .liquidity import (
    LiquidityModel,
    BasicLiquidityModel,
    AdvancedLiquidityModel,
    PerfectLiquidityModel,
    create_liquid_market_model,
    create_illiquid_market_model,
    create_crypto_market_model,
)

__all__ = [
    # Core broker
    "SimulatedBroker",
    "PortfolioStateProtocol",
    "MarketDataProtocol",
    
    # Broker factories
    "create_simulated_broker",
    "create_zero_commission_broker",
    "create_traditional_broker", 
    "create_conservative_broker",
    
    # Slippage models
    "SlippageModel",
    "MarketConditions",
    "PercentageSlippageModel",
    "VolumeImpactSlippageModel",
    "FixedSlippageModel",
    "ZeroSlippageModel",
    
    # Commission models
    "CommissionModel",
    "ZeroCommissionModel",
    "PerShareCommissionModel",
    "PercentageCommissionModel",
    "TieredCommissionModel",
    "FixedCommissionModel",
    "create_alpaca_commission",
    "create_interactive_brokers_commission",
    "create_traditional_broker_commission",
    
    # Liquidity models
    "LiquidityModel",
    "BasicLiquidityModel",
    "AdvancedLiquidityModel",
    "PerfectLiquidityModel",
    "create_liquid_market_model",
    "create_illiquid_market_model",
    "create_crypto_market_model",
]