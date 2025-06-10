"""
Strategy types for signal generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class SignalType(str, Enum):
    """Signal type enumeration."""
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"


class SignalDirection(str, Enum):
    """Signal direction enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # Exit all positions


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    direction: str  # 'long', 'short', 'flat'
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    strategy_id: str  # Strategy that generated the signal
    signal_type: SignalType = SignalType.ENTRY
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event payload."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'signal_type': self.signal_type.value,
            'metadata': self.metadata
        }


@dataclass 
class StrategyConfig:
    """Configuration for a strategy."""
    strategy_id: str
    strategy_type: str  # 'momentum', 'mean_reversion', etc.
    symbols: list[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
