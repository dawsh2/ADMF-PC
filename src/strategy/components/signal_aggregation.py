"""
Signal aggregation components for internal strategy use.

These components help strategies combine signals from multiple internal
sources (indicators, sub-components) into a final trading signal.
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..protocols import SignalDirection


@dataclass
class InternalSignal:
    """Internal signal from a strategy component."""
    source: str  # e.g., "sma_crossover", "rsi_oversold"
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    weight: float = 1.0
    metadata: Dict[str, Any] = None


class SignalCombiner:
    """
    Combines multiple internal signals into a final signal.
    
    This is for intra-strategy use, not multi-strategy aggregation.
    """
    
    def __init__(self, method: str = "weighted_average"):
        self.method = method
        
    def combine(self, signals: List[InternalSignal]) -> Optional[Dict[str, Any]]:
        """Combine internal signals into final signal."""
        if not signals:
            return None
            
        if self.method == "weighted_average":
            return self._weighted_average(signals)
        elif self.method == "majority":
            return self._majority_vote(signals)
        elif self.method == "strongest":
            return self._strongest_signal(signals)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _weighted_average(self, signals: List[InternalSignal]) -> Optional[Dict[str, Any]]:
        """Weighted average of signal strengths."""
        # Group by direction
        buy_strength = 0.0
        sell_strength = 0.0
        total_weight = 0.0
        
        for sig in signals:
            weight = sig.weight * sig.strength
            total_weight += sig.weight
            
            if sig.direction == SignalDirection.BUY:
                buy_strength += weight
            elif sig.direction == SignalDirection.SELL:
                sell_strength += weight
        
        if total_weight == 0:
            return None
            
        # Normalize
        buy_strength /= total_weight
        sell_strength /= total_weight
        
        # Determine final direction
        if buy_strength > sell_strength:
            return {
                'direction': SignalDirection.BUY.value,
                'strength': buy_strength - sell_strength
            }
        elif sell_strength > buy_strength:
            return {
                'direction': SignalDirection.SELL.value,
                'strength': sell_strength - buy_strength
            }
        else:
            return {
                'direction': SignalDirection.HOLD.value,
                'strength': 0.0
            }
    
    def _majority_vote(self, signals: List[InternalSignal]) -> Optional[Dict[str, Any]]:
        """Simple majority voting."""
        buy_votes = sum(1 for s in signals if s.direction == SignalDirection.BUY)
        sell_votes = sum(1 for s in signals if s.direction == SignalDirection.SELL)
        hold_votes = sum(1 for s in signals if s.direction == SignalDirection.HOLD)
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            avg_strength = sum(s.strength for s in signals if s.direction == SignalDirection.BUY) / buy_votes
            return {'direction': SignalDirection.BUY.value, 'strength': avg_strength}
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            avg_strength = sum(s.strength for s in signals if s.direction == SignalDirection.SELL) / sell_votes
            return {'direction': SignalDirection.SELL.value, 'strength': avg_strength}
        else:
            return {'direction': SignalDirection.HOLD.value, 'strength': 0.0}
    
    def _strongest_signal(self, signals: List[InternalSignal]) -> Optional[Dict[str, Any]]:
        """Return the strongest signal."""
        if not signals:
            return None
            
        strongest = max(signals, key=lambda s: s.strength)
        return {
            'direction': strongest.direction.value,
            'strength': strongest.strength
        }


class SignalFilter:
    """
    Filters signals based on various criteria.
    
    Can be used to refine signals before output.
    """
    
    def __init__(self, 
                 min_strength: float = 0.0,
                 required_sources: Optional[List[str]] = None,
                 filter_func: Optional[Callable[[InternalSignal], bool]] = None):
        self.min_strength = min_strength
        self.required_sources = required_sources or []
        self.filter_func = filter_func
    
    def filter(self, signals: List[InternalSignal]) -> List[InternalSignal]:
        """Filter signals based on criteria."""
        filtered = []
        
        for signal in signals:
            # Check strength threshold
            if signal.strength < self.min_strength:
                continue
                
            # Check required sources
            if self.required_sources and signal.source not in self.required_sources:
                continue
                
            # Apply custom filter
            if self.filter_func and not self.filter_func(signal):
                continue
                
            filtered.append(signal)
        
        return filtered