"""
Signal replay components for efficient weight optimization.

These components capture signals during initial optimization runs
and enable efficient replay with different weights, avoiding the
need to re-run full backtests.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from ..protocols import SignalAggregator, SignalDirection


@dataclass
class CapturedSignal:
    """
    Represents a captured signal with full context.
    
    This includes all information needed to replay the signal
    during weight optimization.
    """
    timestamp: datetime
    symbol: str
    source: str  # Which strategy/rule generated it
    direction: SignalDirection
    strength: float
    price: float
    classification: Optional[str] = None  # Market classification at signal time
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['direction'] = self.direction.value if isinstance(self.direction, SignalDirection) else self.direction
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapturedSignal':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data['direction'], str):
            data['direction'] = SignalDirection(data['direction'])
        return cls(**data)


class SignalCapture:
    """
    Captures signals during strategy execution for later replay.
    
    This component intercepts signals and stores them with full
    context including market classification.
    """
    
    def __init__(self, capture_id: str):
        """
        Initialize signal capture.
        
        Args:
            capture_id: Unique identifier for this capture session
        """
        self.capture_id = capture_id
        self.signals: List[CapturedSignal] = []
        self.metadata: Dict[str, Any] = {
            'capture_id': capture_id,
            'start_time': datetime.now(),
            'signal_count': 0
        }
        self._current_classification: Optional[str] = None
        
    def capture_signal(self, 
                      signal: Dict[str, Any],
                      source: str,
                      classification: Optional[str] = None) -> None:
        """
        Capture a signal with context.
        
        Args:
            signal: Signal dictionary
            source: Source strategy/rule name
            classification: Current market classification
        """
        # Use provided classification or current
        if classification is None:
            classification = self._current_classification
            
        captured = CapturedSignal(
            timestamp=signal.get('timestamp', datetime.now()),
            symbol=signal['symbol'],
            source=source,
            direction=signal['direction'],
            strength=signal.get('strength', 1.0),
            price=signal.get('price', 0.0),
            classification=classification,
            metadata=signal.get('metadata', {})
        )
        
        self.signals.append(captured)
        self.metadata['signal_count'] += 1
        
    def update_classification(self, classification: str) -> None:
        """Update current market classification."""
        self._current_classification = classification
        
    def get_signals_by_source(self, source: str) -> List[CapturedSignal]:
        """Get all signals from a specific source."""
        return [s for s in self.signals if s.source == source]
    
    def get_signals_by_classification(self, classification: str) -> List[CapturedSignal]:
        """Get all signals for a specific market classification."""
        return [s for s in self.signals if s.classification == classification]
    
    def save_to_file(self, filepath: str) -> None:
        """Save captured signals to file."""
        data = {
            'metadata': self.metadata,
            'signals': [s.to_dict() for s in self.signals]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SignalCapture':
        """Load captured signals from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        capture = cls(data['metadata']['capture_id'])
        capture.metadata = data['metadata']
        capture.signals = [CapturedSignal.from_dict(s) for s in data['signals']]
        
        return capture


class SignalReplayer:
    """
    Replays captured signals with different weights/parameters.
    
    This enables efficient weight optimization without re-running
    full backtests.
    """
    
    def __init__(self, captured_signals: SignalCapture):
        """
        Initialize signal replayer.
        
        Args:
            captured_signals: Previously captured signals
        """
        self.captured_signals = captured_signals
        self.current_index = 0
        
    def replay_with_weights(self, 
                          weights: Dict[str, float],
                          aggregator: SignalAggregator,
                          classification_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Replay signals with new weights.
        
        Args:
            weights: Dict mapping source names to weights
            aggregator: Signal aggregator to use
            classification_filter: Optional filter for specific classification
            
        Returns:
            List of aggregated signals
        """
        # Group signals by timestamp
        signals_by_time: Dict[datetime, List[CapturedSignal]] = {}
        
        for signal in self.captured_signals.signals:
            # Apply classification filter if specified
            if classification_filter and signal.classification != classification_filter:
                continue
                
            if signal.timestamp not in signals_by_time:
                signals_by_time[signal.timestamp] = []
            signals_by_time[signal.timestamp].append(signal)
        
        # Process each timestamp
        aggregated_signals = []
        
        for timestamp in sorted(signals_by_time.keys()):
            signals_at_time = signals_by_time[timestamp]
            
            # Convert to format expected by aggregator
            weighted_signals = []
            for signal in signals_at_time:
                weight = weights.get(signal.source, 0.0)
                if weight > 0:
                    signal_dict = {
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'strength': signal.strength,
                        'timestamp': signal.timestamp,
                        'price': signal.price,
                        'metadata': signal.metadata
                    }
                    weighted_signals.append((signal_dict, weight))
            
            # Aggregate if we have signals
            if weighted_signals:
                aggregated = aggregator.aggregate(weighted_signals)
                if aggregated:
                    aggregated_signals.append(aggregated)
        
        return aggregated_signals
    
    def calculate_performance(self,
                            weights: Dict[str, float],
                            aggregator: SignalAggregator,
                            performance_calculator: Any,
                            classification: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate performance metrics for given weights.
        
        This is the key method for weight optimization - it can quickly
        evaluate different weight combinations without re-running backtests.
        
        Args:
            weights: Strategy/rule weights
            aggregator: Signal aggregator
            performance_calculator: Component to calculate metrics from signals
            classification: Optional classification filter
            
        Returns:
            Performance metrics
        """
        # Replay signals with weights
        aggregated_signals = self.replay_with_weights(
            weights, aggregator, classification
        )
        
        # Calculate performance from aggregated signals
        # This would typically simulate trades from signals
        return performance_calculator.calculate_from_signals(aggregated_signals)


class WeightedSignalAggregator:
    """
    Aggregates multiple signals using weighted voting.
    
    Implements the SignalAggregator protocol.
    """
    
    def __init__(self, 
                 min_signals: int = 1,
                 min_agreement: float = 0.0):
        """
        Initialize aggregator.
        
        Args:
            min_signals: Minimum signals required
            min_agreement: Minimum agreement ratio (0-1)
        """
        self._min_signals = min_signals
        self.min_agreement = min_agreement
        
    def aggregate(self, signals: List[Tuple[Dict[str, Any], float]]) -> Optional[Dict[str, Any]]:
        """Aggregate multiple weighted signals."""
        if len(signals) < self._min_signals:
            return None
            
        # Separate buy and sell signals
        buy_signals = []
        sell_signals = []
        total_weight = 0.0
        
        for signal, weight in signals:
            total_weight += weight
            
            if signal['direction'] == SignalDirection.BUY:
                buy_signals.append((signal, weight))
            elif signal['direction'] == SignalDirection.SELL:
                sell_signals.append((signal, weight))
        
        if total_weight == 0:
            return None
        
        # Calculate weighted strengths
        buy_strength = sum(s['strength'] * w for s, w in buy_signals)
        sell_strength = sum(s['strength'] * w for s, w in sell_signals)
        
        # Determine direction and strength
        if buy_strength > sell_strength:
            direction = SignalDirection.BUY
            strength = buy_strength / total_weight
            agreement = buy_strength / (buy_strength + sell_strength)
            contributing_signals = buy_signals
        else:
            direction = SignalDirection.SELL
            strength = sell_strength / total_weight
            agreement = sell_strength / (buy_strength + sell_strength)
            contributing_signals = sell_signals
        
        # Check minimum agreement
        if agreement < self.min_agreement:
            return None
        
        # Create aggregated signal
        # Use average price from contributing signals
        avg_price = sum(s['price'] * w for s, w in contributing_signals) / sum(w for _, w in contributing_signals)
        
        return {
            'symbol': signals[0][0]['symbol'],  # Assume all same symbol
            'direction': direction,
            'strength': strength,
            'price': avg_price,
            'timestamp': signals[0][0]['timestamp'],  # Use first timestamp
            'metadata': {
                'aggregation_method': 'weighted',
                'agreement': agreement,
                'signal_count': len(signals),
                'contributing_sources': [s.get('metadata', {}).get('source', 'unknown') for s, _ in contributing_signals]
            }
        }
    
    @property
    def min_signals(self) -> int:
        """Minimum number of signals required."""
        return self._min_signals