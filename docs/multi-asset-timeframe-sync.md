"""
Time synchronization component for multi-symbol, multi-timeframe event-driven backtesting.

This component handles the complex synchronization of bars across different symbols 
and timeframes, ensuring strategies receive complete, aligned data before execution.

Follows Protocol + Composition pattern (no inheritance).
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from ..events import Event, EventType, create_signal_event
from ..containers.protocols import ContainerComponent

logger = logging.getLogger(__name__)


@dataclass
class StrategyDataRequirement:
    """Defines what data a strategy needs."""
    strategy_id: str
    strategy_function: Callable
    classifier_id: Optional[str] = None  # Which classifier gates this strategy
    required_data: List[Tuple[str, str]] = field(default_factory=list)  # [(symbol, timeframe)]
    alignment_mode: str = 'wait_for_all'  # 'wait_for_all' or 'best_effort'
    timeout_ms: int = 5000
    min_bars_required: int = 1  # Minimum bars of history needed


@dataclass 
class TimeAlignmentBuffer(ContainerComponent):
    """
    Component that buffers and aligns bars across multiple symbols and timeframes.
    
    Key responsibilities:
    1. Buffer incoming bars from all symbol/timeframe combinations
    2. Determine when sufficient data is available for each strategy
    3. Trigger feature calculation and strategy execution at the right time
    4. Handle missing data gracefully
    """
    
    # Configuration
    strategy_requirements: List[StrategyDataRequirement] = field(default_factory=list)
    max_buffer_size: int = 1000  # Max bars to buffer per symbol/timeframe
    clock_mode: str = 'market'  # 'market' or 'simulation'
    
    def __post_init__(self):
        # Buffers for incoming bars
        self.bar_buffers: Dict[Tuple[str, str], List[Event]] = defaultdict(list)
        
        # Track latest bar timestamp per symbol/timeframe
        self.latest_timestamps: Dict[Tuple[str, str], datetime] = {}
        
        # Track which data we're waiting for at each time step
        self.pending_data: Dict[datetime, Set[Tuple[str, str]]] = defaultdict(set)
        
        # Group strategies by their data requirements for efficiency
        self.strategy_groups = self._group_strategies_by_requirements()
        
        # Current processing time horizon
        self.current_horizon: Optional[datetime] = None
        
        # Feature cache to avoid recalculation
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        
        # Track strategy execution history
        self.execution_history: Dict[str, datetime] = {}
    
    def initialize(self, container: 'Container') -> None:
        """Initialize with container reference."""
        self.container = container
        
        # Subscribe to BAR events
        self.container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        
        logger.info(f"TimeAlignmentBuffer initialized with {len(self.strategy_requirements)} strategies")
    
    def start(self) -> None:
        """Start the component."""
        logger.info("TimeAlignmentBuffer started")
    
    def stop(self) -> None:
        """Stop the component."""
        logger.info("TimeAlignmentBuffer stopped")
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'buffer_sizes': {f"{sym}_{tf}": len(bars) 
                           for (sym, tf), bars in self.bar_buffers.items()},
            'current_horizon': self.current_horizon.isoformat() if self.current_horizon else None,
            'pending_data_count': sum(len(pending) for pending in self.pending_data.values()),
            'strategy_count': len(self.strategy_requirements)
        }
    
    def on_bar(self, event: Event) -> None:
        """
        Handle incoming bar event.
        
        This is the main entry point that triggers the synchronization logic.
        """
        # Extract bar details
        symbol = event.payload.get('symbol')
        timeframe = event.payload.get('timeframe')
        bar_close_time = event.payload.get('bar_close_time')
        is_complete = event.payload.get('is_complete', True)
        
        if not all([symbol, timeframe, bar_close_time]):
            logger.warning(f"Incomplete bar event: {event.payload}")
            return
        
        # Convert bar_close_time to datetime if needed
        if isinstance(bar_close_time, str):
            bar_close_time = datetime.fromisoformat(bar_close_time)
        
        # Buffer the bar
        key = (symbol, timeframe)
        self.bar_buffers[key].append(event)
        self.latest_timestamps[key] = bar_close_time
        
        # Maintain buffer size
        if len(self.bar_buffers[key]) > self.max_buffer_size:
            self.bar_buffers[key].pop(0)
        
        # Update time horizon
        if self.current_horizon is None or bar_close_time > self.current_horizon:
            self.current_horizon = bar_close_time
        
        # Check if we can process any strategies
        self._check_and_process_strategies(bar_close_time)
    
    def _check_and_process_strategies(self, trigger_time: datetime) -> None:
        """
        Check which strategies can be processed with current data.
        
        This is where the magic happens - determining when we have enough
        synchronized data to run each strategy.
        """
        # Get feature calculator and classifier components
        feature_calc = self.container.get_component('feature_calculator')
        classifier = self.container.get_component('classifier')
        
        if not feature_calc:
            logger.error("No feature calculator component found")
            return
        
        # Check each strategy group
        for group_key, strategies in self.strategy_groups.items():
            required_symbols_timeframes = set(group_key)
            
            # Determine the alignment time for this group
            alignment_time = self._get_alignment_time(required_symbols_timeframes, trigger_time)
            
            if not alignment_time:
                continue
            
            # Check if we have all required data up to alignment_time
            if self._has_required_data(required_symbols_timeframes, alignment_time):
                # Get the aligned bars
                aligned_bars = self._get_aligned_bars(required_symbols_timeframes, alignment_time)
                
                # Calculate features for these bars
                features = self._calculate_features(feature_calc, aligned_bars)
                
                # Get classification if we have a classifier
                classification = None
                if classifier:
                    classification = self._get_classification(classifier, features, aligned_bars)
                
                # Process each strategy in the group
                for strategy_req in strategies:
                    self._process_strategy(strategy_req, features, classification, aligned_bars, alignment_time)
    
    def _get_alignment_time(self, required_data: Set[Tuple[str, str]], trigger_time: datetime) -> Optional[datetime]:
        """
        Determine the appropriate alignment time for a set of required data.
        
        This handles the different bar close times for different timeframes.
        """
        # Find the most restrictive (largest) timeframe
        max_timeframe_minutes = 1
        
        for symbol, timeframe in required_data:
            tf_minutes = self._timeframe_to_minutes(timeframe)
            max_timeframe_minutes = max(max_timeframe_minutes, tf_minutes)
        
        # Align to the largest timeframe boundary
        if max_timeframe_minutes >= 1440:  # Daily
            # Align to market close
            alignment_time = trigger_time.replace(hour=16, minute=0, second=0, microsecond=0)
        elif max_timeframe_minutes >= 60:  # Hourly
            # Align to hour
            alignment_time = trigger_time.replace(minute=0, second=0, microsecond=0)
        elif max_timeframe_minutes >= 5:  # 5-minute or larger
            # Align to timeframe boundary
            minutes = (trigger_time.minute // max_timeframe_minutes) * max_timeframe_minutes
            alignment_time = trigger_time.replace(minute=minutes, second=0, microsecond=0)
        else:  # 1-minute
            # Align to minute
            alignment_time = trigger_time.replace(second=0, microsecond=0)
        
        return alignment_time
    
    def _has_required_data(self, required_data: Set[Tuple[str, str]], alignment_time: datetime) -> bool:
        """Check if we have all required data up to the alignment time."""
        for symbol, timeframe in required_data:
            key = (symbol, timeframe)
            
            # Check if we have any data for this symbol/timeframe
            if key not in self.latest_timestamps:
                return False
            
            # Check if the latest data is recent enough
            expected_bar_time = self._get_expected_bar_time(alignment_time, timeframe)
            if self.latest_timestamps[key] < expected_bar_time:
                return False
        
        return True
    
    def _get_aligned_bars(self, required_data: Set[Tuple[str, str]], alignment_time: datetime) -> Dict[str, Event]:
        """
        Get the most recent bars for each required symbol/timeframe at alignment time.
        
        Returns dict like: {'SPY_1m': Event, 'QQQ_1m': Event, 'NVDA_5m': Event}
        """
        aligned_bars = {}
        
        for symbol, timeframe in required_data:
            key = (symbol, timeframe)
            bars = self.bar_buffers.get(key, [])
            
            if not bars:
                continue
            
            # Find the most recent bar at or before alignment_time
            for bar_event in reversed(bars):
                bar_time = bar_event.payload.get('bar_close_time')
                if isinstance(bar_time, str):
                    bar_time = datetime.fromisoformat(bar_time)
                
                if bar_time <= alignment_time:
                    aligned_bars[f"{symbol}_{timeframe}"] = bar_event
                    break
        
        return aligned_bars
    
    def _calculate_features(self, feature_calc: Any, aligned_bars: Dict[str, Event]) -> Dict[str, Any]:
        """
        Calculate features for the aligned bars.
        
        Uses caching to avoid recalculation when possible.
        """
        # Create cache key from bar timestamps
        cache_key = '_'.join(sorted([
            f"{k}_{b.payload['bar_close_time']}" for k, b in aligned_bars.items()
        ]))
        
        # Check cache
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Calculate features
        features = {}
        for key, bar_event in aligned_bars.items():
            symbol = bar_event.payload['symbol']
            
            # Get symbol-specific features
            if hasattr(feature_calc, 'calculate_features_for_bar'):
                symbol_features = feature_calc.calculate_features_for_bar(bar_event)
                features[symbol] = symbol_features
        
        # Calculate cross-symbol features if needed
        if hasattr(feature_calc, 'calculate_cross_symbol_features'):
            cross_features = feature_calc.calculate_cross_symbol_features(aligned_bars)
            features['cross_symbol'] = cross_features
        
        # Cache the results
        self.feature_cache[cache_key] = features
        
        # Maintain cache size
        if len(self.feature_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self.feature_cache.keys())
            del self.feature_cache[oldest_key]
        
        return features
    
    def _get_classification(self, classifier: Any, features: Dict[str, Any], aligned_bars: Dict[str, Event]) -> Any:
        """Get market classification from classifier."""
        if hasattr(classifier, 'classify'):
            return classifier.classify(features, aligned_bars)
        return None
    
    def _process_strategy(self, strategy_req: StrategyDataRequirement, features: Dict[str, Any], 
                         classification: Any, aligned_bars: Dict[str, Event], alignment_time: datetime) -> None:
        """Process a single strategy with aligned data."""
        # Check if strategy should run based on classifier
        if strategy_req.classifier_id and classification:
            if hasattr(classification, 'classifier_id') and classification.classifier_id != strategy_req.classifier_id:
                return
            # Could also check if classification state allows strategy execution
        
        # Check if we've already processed this strategy for this time
        last_execution = self.execution_history.get(strategy_req.strategy_id)
        if last_execution and last_execution >= alignment_time:
            return
        
        # Call the strategy function
        try:
            signal = strategy_req.strategy_function(features, classification)
            
            if signal:
                # Enrich signal with context
                enriched_signal = self._enrich_signal(signal, strategy_req, classification, aligned_bars, features)
                
                # Publish signal event
                signal_event = create_signal_event(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    strength=signal.get('strength', 1.0),
                    strategy_id=strategy_req.strategy_id,
                    source_id=f"feature_container",
                    container_id=self.container.container_id
                )
                
                # Add enriched payload
                signal_event.payload.update(enriched_signal)
                
                # Publish to parent (root bus)
                self.container.publish_event(signal_event, target_scope="parent")
                
                logger.debug(f"Published signal from {strategy_req.strategy_id} at {alignment_time}")
            
            # Update execution history
            self.execution_history[strategy_req.strategy_id] = alignment_time
            
        except Exception as e:
            logger.error(f"Error processing strategy {strategy_req.strategy_id}: {e}")
    
    def _enrich_signal(self, signal: Dict[str, Any], strategy_req: StrategyDataRequirement,
                      classification: Any, aligned_bars: Dict[str, Event], features: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich signal with context data."""
        enriched = signal.copy()
        
        # Add metadata
        enriched['strategy_id'] = strategy_req.strategy_id
        enriched['classifier_id'] = strategy_req.classifier_id
        
        # Add classification info
        if classification:
            enriched['classification'] = getattr(classification, 'state', str(classification))
        
        # Add bar data (just OHLCV to keep size reasonable)
        enriched['bar_data'] = {}
        for key, bar_event in aligned_bars.items():
            bar_payload = bar_event.payload
            enriched['bar_data'][key] = {
                'open': bar_payload.get('open'),
                'high': bar_payload.get('high'),
                'low': bar_payload.get('low'),
                'close': bar_payload.get('close'),
                'volume': bar_payload.get('volume'),
                'timestamp': bar_payload.get('bar_close_time')
            }
        
        # Add relevant features
        enriched['features'] = features.get(signal['symbol'], {})
        
        return enriched
    
    def _group_strategies_by_requirements(self) -> Dict[Tuple[Tuple[str, str], ...], List[StrategyDataRequirement]]:
        """Group strategies that have identical data requirements."""
        groups = defaultdict(list)
        
        for strategy in self.strategy_requirements:
            # Create a hashable key from required data
            key = tuple(sorted(strategy.required_data))
            groups[key].append(strategy)
        
        return dict(groups)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        conversions = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return conversions.get(timeframe.lower(), 1)
    
    def _get_expected_bar_time(self, current_time: datetime, timeframe: str) -> datetime:
        """Get the expected bar close time for a timeframe at current time."""
        tf_minutes = self._timeframe_to_minutes(timeframe)
        
        if tf_minutes >= 1440:  # Daily
            # Previous market close
            return (current_time - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Previous bar close
            minutes_since_midnight = current_time.hour * 60 + current_time.minute
            last_bar_minutes = (minutes_since_midnight // tf_minutes) * tf_minutes
            
            return current_time.replace(
                hour=last_bar_minutes // 60,
                minute=last_bar_minutes % 60,
                second=0,
                microsecond=0
            )


# Example usage in Feature Container setup

def setup_feature_container_with_synchronization(container: 'Container', config: Dict[str, Any]) -> None:
    """
    Setup a feature container with time synchronization.
    
    Example config:
    {
        'strategies': [
            {
                'id': 'momentum_multi',
                'function': momentum_strategy_func,
                'classifier_id': 'trend_classifier',
                'required_data': [('SPY', '1m'), ('QQQ', '1m'), ('NVDA', '5m')],
                'alignment_mode': 'wait_for_all'
            },
            {
                'id': 'pairs_trade',
                'function': pairs_strategy_func,
                'required_data': [('AAPL', '5m'), ('MSFT', '5m')],
                'alignment_mode': 'wait_for_all'
            }
        ]
    }
    """
    # Create strategy requirements
    strategy_requirements = []
    for strat_config in config.get('strategies', []):
        req = StrategyDataRequirement(
            strategy_id=strat_config['id'],
            strategy_function=strat_config['function'],
            classifier_id=strat_config.get('classifier_id'),
            required_data=strat_config['required_data'],
            alignment_mode=strat_config.get('alignment_mode', 'wait_for_all'),
            timeout_ms=strat_config.get('timeout_ms', 5000)
        )
        strategy_requirements.append(req)
    
    # Create and add the time alignment buffer
    time_buffer = TimeAlignmentBuffer(
        strategy_requirements=strategy_requirements,
        clock_mode=config.get('clock_mode', 'market')
    )
    container.add_component('time_buffer', time_buffer)
    
    logger.info(f"Setup time synchronization for {len(strategy_requirements)} strategies")
