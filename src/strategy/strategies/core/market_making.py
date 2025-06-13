"""
Market making trading strategy.

This demonstrates a pure protocol-based strategy with NO inheritance.
The strategy can be enhanced with capabilities through the ComponentFactory.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ....strategy.protocols import Strategy, SignalDirection


class MarketMakingStrategy:
    """
    Market making strategy for providing liquidity.
    
    This is a simple class with no inheritance. It implements the
    Strategy protocol methods, making it compatible with the system.
    
    Features:
    - Bid-ask spread management
    - Inventory risk management
    - Dynamic spread adjustment
    """
    
    def __init__(self,
                 base_spread: float = 0.001,  # 0.1% base spread
                 max_inventory: float = 1000,  # Maximum inventory per side
                 inventory_skew_factor: float = 0.5,  # How much to skew prices based on inventory
                 spread_volatility_mult: float = 2.0):  # Spread multiplier during high volatility
        """Initialize market making strategy."""
        # Parameters
        self.base_spread = base_spread
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self.spread_volatility_mult = spread_volatility_mult
        
        # State
        self.current_inventory: Dict[str, float] = {}  # symbol -> inventory
        self.price_history: Dict[str, List[float]] = {}  # symbol -> prices
        self.last_quotes: Dict[str, Dict[str, float]] = {}  # symbol -> {bid, ask}
        self.volatility_estimate: Dict[str, float] = {}  # symbol -> volatility
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "market_making_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal from market data.
        
        For now, this is a placeholder implementation.
        Full market making logic would include:
        - Continuous quoting on both sides
        - Inventory management
        - Adverse selection protection
        - Dynamic spread adjustment
        """
        # Extract basic data
        symbol = market_data.get('symbol', 'UNKNOWN')
        mid_price = market_data.get('close', market_data.get('price'))
        timestamp = market_data.get('timestamp', datetime.now())
        
        if mid_price is None:
            return None
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(mid_price)
        
        # Limit history to last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol].pop(0)
        
        # Calculate simple volatility estimate
        if len(self.price_history[symbol]) >= 20:
            recent_prices = self.price_history[symbol][-20:]
            returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                      for i in range(1, len(recent_prices))]
            volatility = sum(abs(r) for r in returns) / len(returns)
            self.volatility_estimate[symbol] = volatility
        else:
            volatility = self.base_spread
        
        # TODO: Implement actual market making logic
        # This would involve:
        # 1. Calculating optimal bid/ask quotes
        # 2. Managing inventory risk
        # 3. Adjusting spreads based on market conditions
        # 4. Handling order execution and updates
        
        # For now, return None (no signal)
        return None
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.current_inventory.clear()
        self.price_history.clear()
        self.last_quotes.clear()
        self.volatility_estimate.clear()
    
    # Note: Optimization methods are added by OptimizationCapability
    # when the strategy is created through ComponentFactory.


# Example of creating the strategy with capabilities
def create_market_making_strategy(config: Dict[str, Any] = None) -> Any:
    """
    Factory function to create market making strategy with capabilities.
    
    This would typically use ComponentFactory to add capabilities.
    """
    # Default configuration
    default_config = {
        'base_spread': 0.001,
        'max_inventory': 1000,
        'inventory_skew_factor': 0.5,
        'spread_volatility_mult': 2.0
    }
    
    if config:
        default_config.update(config)
    
    # Create strategy instance
    strategy = MarketMakingStrategy(**default_config)
    
    # In real usage, this would use ComponentFactory
    
    return strategy