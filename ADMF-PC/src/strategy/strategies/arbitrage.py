"""
Arbitrage strategy placeholder.
"""

from ..base import StrategyBase


class ArbitrageStrategy(StrategyBase):
    """Arbitrage strategy - to be implemented."""
    
    def __init__(self, name: str = "arbitrage_strategy"):
        super().__init__(name)
        # TODO: Implement arbitrage logic
    
    def on_bar(self, event):
        """Process bar event."""
        pass