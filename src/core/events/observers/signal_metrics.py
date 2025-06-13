"""
Signal-only metrics observer for signal generation workflows.

Tracks signal quality, frequency, and characteristics without requiring trades or fills.
Designed for signal generation validation and analysis.
Now includes signal performance calculations using entry/exit prices.
"""

from typing import Dict, Any, Optional, List, DefaultDict
from collections import defaultdict, deque
from datetime import datetime
import logging
import statistics

from ..protocols import EventObserverProtocol
from ..types import Event, EventType
from .signal_performance_calculator import SignalPairMatcher, SignalOnlyPerformance

logger = logging.getLogger(__name__)


class SignalMetricsCalculator:
    """
    Calculates signal-specific metrics during signal generation.
    
    Tracks signal quality, frequency, timing, and strategy performance
    without requiring trade execution.
    """
    
    def __init__(self):
        """Initialize signal metrics calculator."""
        # Signal counts
        self.total_signals = 0
        self.long_signals = 0
        self.short_signals = 0
        self.signals_per_strategy = defaultdict(int)
        
        # Signal characteristics
        self.signal_strengths = []
        self.signal_timings = []  # Time between bar and signal
        
        # Bar processing
        self.total_bars = 0
        self.bars_with_signals = 0
        self.first_bar_time = None
        self.last_bar_time = None
        
        # Strategy analysis
        self.strategy_signals = defaultdict(list)  # strategy -> [signals]
        self.strategy_performance = defaultdict(dict)
        
        # Feature tracking
        self.feature_coverage = defaultdict(int)  # feature -> count
        self.feature_errors = 0
        
        # Timing analysis
        self.bar_timestamps = []
        self.signal_timestamps = []
        
        # Signal performance tracking
        self.signal_matcher = SignalPairMatcher()
        self.performance_calc = SignalOnlyPerformance()
        
    def process_bar_event(self, event: Event) -> None:
        """Process a BAR event for metrics."""
        self.total_bars += 1
        
        bar_time = event.payload.get('timestamp')
        if bar_time:
            if self.first_bar_time is None:
                self.first_bar_time = bar_time
            self.last_bar_time = bar_time
            self.bar_timestamps.append(bar_time)
    
    def process_signal_event(self, event: Event) -> None:
        """Process a SIGNAL event for metrics."""
        self.total_signals += 1
        
        payload = event.payload
        signal_type = payload.get('signal_type', 'UNKNOWN')
        signal_strength = payload.get('signal_strength', 0.0)
        strategy_name = payload.get('strategy_name', 'default')
        
        # Count by type
        if signal_type == 'LONG':
            self.long_signals += 1
        elif signal_type == 'SHORT':
            self.short_signals += 1
            
        # Track strength
        if signal_strength:
            self.signal_strengths.append(signal_strength)
            
        # Track by strategy
        self.signals_per_strategy[strategy_name] += 1
        self.strategy_signals[strategy_name].append({
            'type': signal_type,
            'strength': signal_strength,
            'timestamp': event.timestamp
        })
        
        # Track timing
        signal_time = event.timestamp
        if signal_time:
            self.signal_timestamps.append(signal_time)
        
        # Process signal for performance tracking
        # Convert to format expected by signal matcher
        signal_data = {
            'strategy_name': strategy_name,
            'symbol': payload.get('symbol', 'UNKNOWN'),
            'signal_type': payload.get('signal_type', 'entry'),  # Map to entry/exit
            'direction': payload.get('direction', 'long'),
            'price': payload.get('price', 0),
            'timestamp': event.timestamp
        }
        
        # Process through matcher
        completed_pair = self.signal_matcher.process_signal(signal_data, event.timestamp)
        
        # If we completed a pair, calculate performance
        if completed_pair:
            trade_result = self.performance_calc.process_signal_pair(completed_pair)
            logger.debug(f"Completed signal pair for {strategy_name}: P&L = {trade_result.get('pnl_pct', 0):.2%}")
    
    def process_feature_event(self, event: Event) -> None:
        """Process feature calculation events."""
        payload = event.payload
        features = payload.get('features', {})
        
        for feature_name, value in features.items():
            if value is not None:
                self.feature_coverage[feature_name] += 1
            else:
                self.feature_errors += 1
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive signal metrics."""
        metrics = {}
        
        # Basic counts
        metrics['signal_counts'] = {
            'total_signals': self.total_signals,
            'long_signals': self.long_signals,
            'short_signals': self.short_signals,
            'signals_per_strategy': dict(self.signals_per_strategy)
        }
        
        # Signal frequency
        if self.total_bars > 0:
            metrics['signal_frequency'] = {
                'signals_per_bar': self.total_signals / self.total_bars,
                'bars_with_signals': self.bars_with_signals,
                'signal_rate': self.bars_with_signals / self.total_bars if self.total_bars > 0 else 0
            }
        
        # Signal strength analysis
        if self.signal_strengths:
            metrics['signal_strength'] = {
                'avg_strength': statistics.mean(self.signal_strengths),
                'median_strength': statistics.median(self.signal_strengths),
                'min_strength': min(self.signal_strengths),
                'max_strength': max(self.signal_strengths),
                'strength_std': statistics.stdev(self.signal_strengths) if len(self.signal_strengths) > 1 else 0
            }
        
        # Strategy analysis
        if self.strategy_signals:
            metrics['strategy_analysis'] = {}
            for strategy, signals in self.strategy_signals.items():
                strategy_metrics = {
                    'signal_count': len(signals),
                    'long_count': sum(1 for s in signals if s['type'] == 'LONG'),
                    'short_count': sum(1 for s in signals if s['type'] == 'SHORT'),
                }
                
                strengths = [s['strength'] for s in signals if s['strength']]
                if strengths:
                    strategy_metrics['avg_strength'] = statistics.mean(strengths)
                    
                metrics['strategy_analysis'][strategy] = strategy_metrics
        
        # Data quality
        metrics['data_quality'] = {
            'total_bars': self.total_bars,
            'feature_coverage': dict(self.feature_coverage),
            'feature_errors': self.feature_errors,
            'data_completeness': (1 - self.feature_errors / max(1, self.total_bars))
        }
        
        # Timing analysis
        if self.first_bar_time and self.last_bar_time:
            duration = self.last_bar_time - self.first_bar_time if isinstance(self.last_bar_time, datetime) else None
            metrics['timing'] = {
                'first_bar': str(self.first_bar_time),
                'last_bar': str(self.last_bar_time),
                'duration': str(duration) if duration else None,
                'total_bars': self.total_bars
            }
        
        # Add signal matching statistics
        metrics['signal_matching'] = self.signal_matcher.get_statistics()
        
        # Add performance metrics
        metrics['performance'] = self.performance_calc.get_metrics()
        
        return metrics


class SignalMetricsObserver:
    """
    Event observer for signal-only metrics collection.
    
    Designed for signal generation workflows where no trades are executed.
    Focuses on signal quality, frequency, and strategy analysis.
    """
    
    def __init__(self, calculator: Optional[SignalMetricsCalculator] = None):
        """
        Initialize signal metrics observer.
        
        Args:
            calculator: Signal metrics calculator (creates default if None)
        """
        self.calculator = calculator or SignalMetricsCalculator()
        self.events_processed = 0
        self.start_time = datetime.now()
        
    def on_event(self, event: Event) -> None:
        """Process event for signal metrics."""
        self.events_processed += 1
        
        event_type = getattr(event, 'event_type', None)
        
        if event_type == 'BAR':
            self.calculator.process_bar_event(event)
            
        elif event_type == 'SIGNAL':
            self.calculator.process_signal_event(event)
            
        elif event_type == 'FEATURE':
            self.calculator.process_feature_event(event)
            
        # Log periodic updates
        if self.events_processed % 100 == 0:
            logger.debug(f"Processed {self.events_processed} events, "
                        f"{self.calculator.total_signals} signals generated")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current signal metrics."""
        metrics = self.calculator.calculate_metrics()
        
        # Add observer stats
        metrics['observer_stats'] = {
            'events_processed': self.events_processed,
            'processing_time': (datetime.now() - self.start_time).total_seconds(),
            'events_per_second': self.events_processed / max(1, (datetime.now() - self.start_time).total_seconds())
        }
        
        return metrics
    
    def get_results(self) -> Dict[str, Any]:
        """Get final results (alias for get_metrics for consistency)."""
        return self.get_metrics()
    
    def reset(self) -> None:
        """Reset metrics for new run."""
        self.calculator = SignalMetricsCalculator()
        self.events_processed = 0
        self.start_time = datetime.now()


# Convenience function for easy integration
def create_signal_metrics_observer() -> SignalMetricsObserver:
    """Create a signal metrics observer with default calculator."""
    return SignalMetricsObserver()