"""
Signal generation rules.
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

from ..protocols import Rule


logger = logging.getLogger(__name__)


class SignalRule(ABC):
    """Base class for signal generation rules."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_signal: Optional[Dict[str, Any]] = None
        self.signal_count = 0
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate rule and generate signal if conditions are met."""
        pass
    
    def reset(self) -> None:
        """Reset rule state."""
        self.last_signal = None
        self.signal_count = 0


class ThresholdRule(SignalRule):
    """
    Generates signal when a value crosses a threshold.
    """
    
    def __init__(self, 
                 name: str,
                 value_key: str,
                 threshold: float,
                 direction: str = "above",  # above, below
                 signal_type: str = "BUY"):
        super().__init__(name)
        self.value_key = value_key
        self.threshold = threshold
        self.direction = direction
        self.signal_type = signal_type
        self.was_triggered = False
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if value crosses threshold."""
        value = data.get(self.value_key)
        
        if value is None:
            return None
        
        # Check threshold condition
        if self.direction == "above":
            is_triggered = value > self.threshold
        else:
            is_triggered = value < self.threshold
        
        # Generate signal on transition
        if is_triggered and not self.was_triggered:
            signal = {
                'type': self.signal_type,
                'rule': self.name,
                'value': value,
                'threshold': self.threshold,
                'confidence': abs(value - self.threshold) / self.threshold
            }
            self.last_signal = signal
            self.signal_count += 1
            self.was_triggered = True
            return signal
        elif not is_triggered:
            self.was_triggered = False
        
        return None
    
    def reset(self) -> None:
        """Reset rule state."""
        super().reset()
        self.was_triggered = False


class CrossoverRule(SignalRule):
    """
    Generates signal when two values cross.
    """
    
    def __init__(self,
                 name: str,
                 fast_key: str,
                 slow_key: str,
                 signal_on: str = "bullish",  # bullish, bearish, both
                 min_separation: float = 0.0):
        super().__init__(name)
        self.fast_key = fast_key
        self.slow_key = slow_key
        self.signal_on = signal_on
        self.min_separation = min_separation
        
        self.prev_fast: Optional[float] = None
        self.prev_slow: Optional[float] = None
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for crossover."""
        fast = data.get(self.fast_key)
        slow = data.get(self.slow_key)
        
        if fast is None or slow is None:
            return None
        
        signal = None
        
        # Check for crossover
        if self.prev_fast is not None and self.prev_slow is not None:
            # Bullish crossover (fast crosses above slow)
            if (self.prev_fast <= self.prev_slow and fast > slow and 
                abs(fast - slow) >= self.min_separation):
                
                if self.signal_on in ["bullish", "both"]:
                    signal = {
                        'type': 'BUY',
                        'rule': self.name,
                        'fast_value': fast,
                        'slow_value': slow,
                        'crossover': 'bullish',
                        'strength': abs(fast - slow) / slow if slow != 0 else 0
                    }
            
            # Bearish crossover (fast crosses below slow)
            elif (self.prev_fast >= self.prev_slow and fast < slow and
                  abs(fast - slow) >= self.min_separation):
                
                if self.signal_on in ["bearish", "both"]:
                    signal = {
                        'type': 'SELL',
                        'rule': self.name,
                        'fast_value': fast,
                        'slow_value': slow,
                        'crossover': 'bearish',
                        'strength': abs(fast - slow) / slow if slow != 0 else 0
                    }
        
        # Update previous values
        self.prev_fast = fast
        self.prev_slow = slow
        
        if signal:
            self.last_signal = signal
            self.signal_count += 1
            return signal
        
        return None
    
    def reset(self) -> None:
        """Reset rule state."""
        super().reset()
        self.prev_fast = None
        self.prev_slow = None


class PatternRule(SignalRule):
    """
    Generates signal when a specific pattern is detected.
    """
    
    def __init__(self,
                 name: str,
                 pattern_detector: Callable[[Dict[str, Any]], bool],
                 signal_type: str = "BUY",
                 confidence_calculator: Optional[Callable[[Dict[str, Any]], float]] = None):
        super().__init__(name)
        self.pattern_detector = pattern_detector
        self.signal_type = signal_type
        self.confidence_calculator = confidence_calculator
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for pattern."""
        if self.pattern_detector(data):
            signal = {
                'type': self.signal_type,
                'rule': self.name,
                'pattern': self.name
            }
            
            # Calculate confidence if calculator provided
            if self.confidence_calculator:
                signal['confidence'] = self.confidence_calculator(data)
            else:
                signal['confidence'] = 0.5
            
            self.last_signal = signal
            self.signal_count += 1
            return signal
        
        return None


class CompositeRule(SignalRule):
    """
    Combines multiple rules with configurable logic.
    """
    
    def __init__(self,
                 name: str,
                 rules: List[SignalRule],
                 logic: str = "all",  # all, any, majority
                 min_rules: int = 1):
        super().__init__(name)
        self.rules = rules
        self.logic = logic
        self.min_rules = min_rules
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate all rules and combine signals."""
        signals = []
        
        # Evaluate all sub-rules
        for rule in self.rules:
            signal = rule.evaluate(data)
            if signal:
                signals.append(signal)
        
        # Apply combination logic
        if self.logic == "all":
            if len(signals) == len(self.rules):
                return self._combine_signals(signals)
        
        elif self.logic == "any":
            if len(signals) >= self.min_rules:
                return self._combine_signals(signals)
        
        elif self.logic == "majority":
            if len(signals) > len(self.rules) / 2:
                return self._combine_signals(signals)
        
        return None
    
    def _combine_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple signals into one."""
        # Determine signal type (majority vote)
        buy_count = sum(1 for s in signals if s['type'] == 'BUY')
        sell_count = sum(1 for s in signals if s['type'] == 'SELL')
        
        signal_type = 'BUY' if buy_count >= sell_count else 'SELL'
        
        # Average confidence
        confidences = [s.get('confidence', 0.5) for s in signals]
        avg_confidence = sum(confidences) / len(confidences)
        
        combined_signal = {
            'type': signal_type,
            'rule': self.name,
            'sub_signals': signals,
            'confidence': avg_confidence,
            'agreement': len(signals) / len(self.rules)
        }
        
        self.last_signal = combined_signal
        self.signal_count += 1
        
        return combined_signal
    
    def reset(self) -> None:
        """Reset all rules."""
        super().reset()
        for rule in self.rules:
            rule.reset()