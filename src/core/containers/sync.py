"""
Unified time synchronization system for multi-asset, multi-timeframe backtesting.

This module provides core synchronization components that handle:
- Bar buffering and alignment across symbols/timeframes
- Strategy orchestration with data requirements
- Feature calculation and caching
- Classifier-aware execution
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import logging

from ..events import Event, EventType
from ..types.events import create_signal_event
from .protocols import ContainerComponent

logger = logging.getLogger(__name__)


class AlignmentMode(Enum):
    """How to handle data alignment."""
    ALL = 'all'  # Wait for all required data
    ANY = 'any'  # Process when any subset is ready
    BEST_EFFORT = 'best_effort'  # Process with whatever is available
    ROLLING = 'rolling'  # Rolling window alignment


class TimeframeAlignment(Enum):
    """How to align different timeframes."""
    STRICT = 'strict'  # All bars must align perfectly
    NEAREST = 'nearest'  # Use nearest available bar
    FORWARD_FILL = 'forward_fill'  # Use last known value


@dataclass
class DataRequirement:
    """Specifies data needed by a strategy or component."""
    symbols: List[str]
    timeframes: List[str]
    min_history: int = 1  # Minimum bars of history needed
    alignment_mode: AlignmentMode = AlignmentMode.ALL
    timeframe_alignment: TimeframeAlignment = TimeframeAlignment.NEAREST
    timeout_ms: Optional[int] = None  # Max wait time for data
    
    def __post_init__(self):
        # Create all combinations
        self.required_keys = [
            (symbol, timeframe) 
            for symbol in self.symbols 
            for timeframe in self.timeframes
        ]
    
    def to_key(self) -> Tuple[Tuple[str, str], ...]:
        """Create hashable key for grouping."""
        return tuple(sorted(self.required_keys))


@dataclass
class StrategySpecification:
    """Complete specification for a strategy."""
    strategy_id: str
    strategy_function: Callable
    data_requirement: DataRequirement
    classifier_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    # Execution constraints
    min_time_between_signals: Optional[timedelta] = None
    max_signals_per_day: Optional[int] = None
    allowed_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour)


@dataclass
class BarBuffer:
    """Efficient buffer for bar data with history management."""
    symbol: str
    timeframe: str
    max_size: int = 1000
    
    def __post_init__(self):
        self.bars: List[Event] = []
        self.timestamps: List[datetime] = []
        self.latest_time: Optional[datetime] = None
        
    def add(self, event: Event) -> None:
        """Add bar to buffer maintaining time order."""
        bar_time = self._extract_time(event)
        
        # Find insertion point to maintain order
        insert_idx = len(self.bars)
        for i in range(len(self.bars) - 1, -1, -1):
            if self.timestamps[i] <= bar_time:
                insert_idx = i + 1
                break
        
        # Insert maintaining order
        self.bars.insert(insert_idx, event)
        self.timestamps.insert(insert_idx, bar_time)
        
        # Update latest
        if self.latest_time is None or bar_time > self.latest_time:
            self.latest_time = bar_time
        
        # Maintain size limit
        if len(self.bars) > self.max_size:
            self.bars.pop(0)
            self.timestamps.pop(0)
    
    def get_at_time(self, target_time: datetime, 
                    mode: TimeframeAlignment = TimeframeAlignment.NEAREST) -> Optional[Event]:
        """Get bar at or near target time."""
        if not self.bars:
            return None
        
        if mode == TimeframeAlignment.STRICT:
            # Exact match only
            for i, ts in enumerate(self.timestamps):
                if ts == target_time:
                    return self.bars[i]
            return None
            
        elif mode == TimeframeAlignment.NEAREST:
            # Find closest bar
            best_idx = 0
            best_diff = abs((self.timestamps[0] - target_time).total_seconds())
            
            for i, ts in enumerate(self.timestamps[1:], 1):
                diff = abs((ts - target_time).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            
            return self.bars[best_idx]
            
        else:  # FORWARD_FILL
            # Most recent bar before target
            for i in range(len(self.timestamps) - 1, -1, -1):
                if self.timestamps[i] <= target_time:
                    return self.bars[i]
            return None
    
    def get_history(self, count: int) -> List[Event]:
        """Get last N bars."""
        return self.bars[-count:] if count <= len(self.bars) else self.bars.copy()
    
    def _extract_time(self, event: Event) -> datetime:
        """Extract timestamp from event."""
        bar_time = event.payload.get('bar_close_time')
        if isinstance(bar_time, str):
            bar_time = datetime.fromisoformat(bar_time)
        return bar_time


@dataclass
class TimeAlignmentBuffer(ContainerComponent):
    """
    Core synchronization component that buffers and aligns bars.
    
    This is the pure synchronization logic, separated from orchestration.
    """
    
    # Configuration
    buffer_size: int = 1000
    default_alignment: TimeframeAlignment = TimeframeAlignment.NEAREST
    emit_partial: bool = False  # Emit events even with missing data
    
    def __post_init__(self):
        # Buffers indexed by (symbol, timeframe)
        self.buffers: Dict[Tuple[str, str], BarBuffer] = {}
        
        # Track what we're synchronizing
        self.tracked_keys: Set[Tuple[str, str]] = set()
        
        # Synchronization state
        self.last_sync_time: Optional[datetime] = None
        self.pending_syncs: List[datetime] = []
        
        # Callback for synchronized data
        self.sync_callbacks: List[Callable] = []
        
    def initialize(self, container: 'Container') -> None:
        """Initialize with container reference."""
        self.container = container
        
        # Subscribe to BAR events
        self.container.event_bus.subscribe(EventType.BAR.value, self.on_bar)
        
        logger.info(f"TimeAlignmentBuffer initialized with buffer_size={self.buffer_size}")
    
    def start(self) -> None:
        """Start the component."""
        logger.info("TimeAlignmentBuffer started")
    
    def stop(self) -> None:
        """Stop and report final state."""
        buffer_summary = {
            f"{s}_{t}": len(b.bars) 
            for (s, t), b in self.buffers.items()
        }
        logger.info(f"TimeAlignmentBuffer stopped. Final buffers: {buffer_summary}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'buffer_count': len(self.buffers),
            'total_bars': sum(len(b.bars) for b in self.buffers.values()),
            'tracked_keys': list(self.tracked_keys),
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None
        }
    
    def track_symbols_timeframes(self, requirements: List[DataRequirement]) -> None:
        """Register which symbol/timeframe combinations to track."""
        for req in requirements:
            for key in req.required_keys:
                self.tracked_keys.add(key)
                if key not in self.buffers:
                    self.buffers[key] = BarBuffer(
                        symbol=key[0],
                        timeframe=key[1],
                        max_size=self.buffer_size
                    )
        
        logger.info(f"Tracking {len(self.tracked_keys)} symbol/timeframe combinations")
    
    def register_callback(self, callback: Callable) -> None:
        """Register callback for synchronized data."""
        self.sync_callbacks.append(callback)
    
    def on_bar(self, event: Event) -> None:
        """Handle incoming bar event."""
        symbol = event.payload.get('symbol')
        timeframe = event.payload.get('timeframe')
        key = (symbol, timeframe)
        
        # Only process tracked combinations
        if key not in self.tracked_keys:
            return
        
        # Add to buffer
        self.buffers[key].add(event)
        
        # Extract time and check for synchronization opportunity
        bar_time = self._extract_bar_time(event)
        self._check_synchronization(bar_time)
    
    def _check_synchronization(self, trigger_time: datetime) -> None:
        """Check if we can perform synchronization."""
        # Determine sync points based on timeframes we're tracking
        sync_times = self._get_sync_times(trigger_time)
        
        for sync_time in sync_times:
            if sync_time <= trigger_time and sync_time not in self.pending_syncs:
                self.pending_syncs.append(sync_time)
        
        # Process pending syncs
        while self.pending_syncs:
            sync_time = self.pending_syncs.pop(0)
            self._perform_synchronization(sync_time)
    
    def _get_sync_times(self, current_time: datetime) -> List[datetime]:
        """Determine potential synchronization times."""
        sync_times = []
        
        # Get unique timeframes
        timeframes = set(tf for _, tf in self.tracked_keys)
        
        for tf in timeframes:
            # Calculate the sync time for this timeframe
            sync_time = self._align_to_timeframe(current_time, tf)
            if sync_time not in sync_times:
                sync_times.append(sync_time)
        
        return sorted(sync_times)
    
    def _perform_synchronization(self, sync_time: datetime) -> None:
        """Perform synchronization at specified time."""
        # Don't re-sync same time
        if self.last_sync_time and sync_time <= self.last_sync_time:
            return
        
        # Collect aligned bars
        aligned_data = self.get_aligned_bars(
            list(self.tracked_keys),
            sync_time,
            self.default_alignment
        )
        
        # Check if we have enough data
        if not aligned_data or (not self.emit_partial and len(aligned_data) < len(self.tracked_keys)):
            return
        
        # Update sync time
        self.last_sync_time = sync_time
        
        # Notify callbacks
        for callback in self.sync_callbacks:
            try:
                callback(aligned_data, sync_time)
            except Exception as e:
                logger.error(f"Error in sync callback: {e}")
        
        # Emit synchronized event
        self._emit_synchronized_event(aligned_data, sync_time)
    
    def get_aligned_bars(self, 
                        keys: List[Tuple[str, str]], 
                        target_time: datetime,
                        alignment: TimeframeAlignment = TimeframeAlignment.NEAREST) -> Dict[str, Event]:
        """Get aligned bars for specified keys at target time."""
        aligned = {}
        
        for symbol, timeframe in keys:
            key = (symbol, timeframe)
            buffer = self.buffers.get(key)
            
            if buffer:
                bar = buffer.get_at_time(target_time, alignment)
                if bar:
                    aligned[f"{symbol}_{timeframe}"] = bar
        
        return aligned
    
    def get_history(self, key: Tuple[str, str], count: int) -> List[Event]:
        """Get historical bars for a specific symbol/timeframe."""
        buffer = self.buffers.get(key)
        return buffer.get_history(count) if buffer else []
    
    def _emit_synchronized_event(self, aligned_data: Dict[str, Event], sync_time: datetime) -> None:
        """Emit event with synchronized data."""
        # Extract just the bar data
        bars = {}
        features = {}
        
        for key, event in aligned_data.items():
            bars[key] = event.payload.get('bar')
            if 'features' in event.payload:
                features[key] = event.payload['features']
        
        # Create synchronized event
        sync_event = Event(
            event_type='SYNCHRONIZED_BARS',
            payload={
                'bars': bars,
                'features': features,
                'sync_time': sync_time,
                'symbols': list(set(k.split('_')[0] for k in aligned_data.keys())),
                'timeframes': list(set(k.split('_')[1] for k in aligned_data.keys())),
                'alignment_mode': self.default_alignment.value
            },
            source_id=self.container.container_id if self.container else 'time_buffer'
        )
        
        if self.container:
            self.container.event_bus.publish(sync_event)
    
    def _align_to_timeframe(self, time: datetime, timeframe: str) -> datetime:
        """Align time to timeframe boundary."""
        tf_minutes = self._timeframe_to_minutes(timeframe)
        
        if tf_minutes >= 1440:  # Daily
            return time.replace(hour=16, minute=0, second=0, microsecond=0)
        elif tf_minutes >= 60:  # Hourly
            return time.replace(minute=0, second=0, microsecond=0)
        else:  # Sub-hourly
            minutes = (time.minute // tf_minutes) * tf_minutes
            return time.replace(minute=minutes, second=0, microsecond=0)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        conversions = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return conversions.get(timeframe.lower(), 1)
    
    def _extract_bar_time(self, event: Event) -> datetime:
        """Extract timestamp from bar event."""
        bar_time = event.payload.get('bar_close_time')
        if isinstance(bar_time, str):
            bar_time = datetime.fromisoformat(bar_time)
        return bar_time


@dataclass
class StrategyOrchestrator(ContainerComponent):
    """
    Orchestrates strategy execution using synchronized data.
    
    This component:
    - Manages strategy specifications and requirements
    - Coordinates with TimeAlignmentBuffer for data
    - Handles feature calculation and caching
    - Manages classifier integration
    - Tracks execution history
    - Enriches and publishes signals
    """
    
    # Configuration
    strategies: List[StrategySpecification] = field(default_factory=list)
    feature_cache_size: int = 1000
    enable_execution_tracking: bool = True
    
    def __post_init__(self):
        # Group strategies by data requirements for efficiency
        self.strategy_groups: Dict[Tuple, List[StrategySpecification]] = defaultdict(list)
        self._group_strategies()
        
        # Caching
        self.feature_cache: Dict[str, Any] = {}
        self.classifier_cache: Dict[str, Any] = {}
        
        # Execution tracking
        self.execution_history: Dict[str, List[datetime]] = defaultdict(list)
        self.signal_counts: Dict[str, int] = defaultdict(int)
        
        # Component references
        self.time_buffer: Optional[TimeAlignmentBuffer] = None
        self.feature_calculator: Optional[Any] = None
        self.classifier: Optional[Any] = None
        
    def initialize(self, container: 'Container') -> None:
        """Initialize and setup components."""
        self.container = container
        
        # Get required components
        self.time_buffer = container.get_component('time_buffer')
        self.feature_calculator = container.get_component('feature_calculator')
        self.classifier = container.get_component('classifier')
        
        if not self.time_buffer:
            raise ValueError("StrategyOrchestrator requires TimeAlignmentBuffer component")
        
        # Register data requirements with time buffer
        requirements = [s.data_requirement for s in self.strategies if s.enabled]
        self.time_buffer.track_symbols_timeframes(requirements)
        
        # Register for synchronized data
        self.time_buffer.register_callback(self.on_synchronized_data)
        
        logger.info(f"StrategyOrchestrator initialized with {len(self.strategies)} strategies")
    
    def start(self) -> None:
        """Start the orchestrator."""
        logger.info("StrategyOrchestrator started")
    
    def stop(self) -> None:
        """Stop and report statistics."""
        stats = {
            'total_signals': sum(self.signal_counts.values()),
            'by_strategy': dict(self.signal_counts),
            'cache_size': len(self.feature_cache)
        }
        logger.info(f"StrategyOrchestrator stopped. Stats: {stats}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get orchestrator state."""
        return {
            'active_strategies': sum(1 for s in self.strategies if s.enabled),
            'total_signals': sum(self.signal_counts.values()),
            'feature_cache_size': len(self.feature_cache),
            'execution_tracking': self.enable_execution_tracking
        }
    
    def on_synchronized_data(self, aligned_data: Dict[str, Event], sync_time: datetime) -> None:
        """Process synchronized data through strategies."""
        # Process each strategy group
        for group_key, strategies in self.strategy_groups.items():
            # Check if this group has required data
            if self._has_required_data(strategies[0].data_requirement, aligned_data):
                self._process_strategy_group(strategies, aligned_data, sync_time)
    
    def _process_strategy_group(self, 
                               strategies: List[StrategySpecification],
                               aligned_data: Dict[str, Event],
                               sync_time: datetime) -> None:
        """Process a group of strategies with same data requirements."""
        # Calculate features (with caching)
        features = self._calculate_features(aligned_data, sync_time)
        
        # Get classification if needed
        classification = None
        if self.classifier and any(s.classifier_id for s in strategies):
            classification = self._get_classification(features, aligned_data, sync_time)
        
        # Process each strategy
        for strategy in strategies:
            if strategy.enabled:
                self._execute_strategy(strategy, features, classification, aligned_data, sync_time)
    
    def _execute_strategy(self,
                         strategy: StrategySpecification,
                         features: Dict[str, Any],
                         classification: Optional[Any],
                         aligned_data: Dict[str, Event],
                         sync_time: datetime) -> None:
        """Execute a single strategy."""
        # Check execution constraints
        if not self._check_execution_allowed(strategy, sync_time):
            return
        
        # Check classifier gate
        if strategy.classifier_id and classification:
            if not self._check_classifier_gate(strategy.classifier_id, classification):
                return
        
        try:
            # Prepare strategy inputs
            strategy_features = self._prepare_strategy_features(strategy, features)
            
            # Call strategy function
            signal = strategy.strategy_function(
                features=strategy_features,
                classification=classification,
                parameters=strategy.parameters
            )
            
            if signal and self._validate_signal(signal):
                # Enrich and publish signal
                self._process_signal(signal, strategy, features, classification, aligned_data, sync_time)
                
                # Update tracking
                if self.enable_execution_tracking:
                    self.execution_history[strategy.strategy_id].append(sync_time)
                self.signal_counts[strategy.strategy_id] += 1
        
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.strategy_id}: {e}", exc_info=True)
    
    def _check_execution_allowed(self, strategy: StrategySpecification, current_time: datetime) -> bool:
        """Check if strategy execution is allowed based on constraints."""
        # Check time between signals
        if strategy.min_time_between_signals and self.execution_history.get(strategy.strategy_id):
            last_execution = self.execution_history[strategy.strategy_id][-1]
            if current_time - last_execution < strategy.min_time_between_signals:
                return False
        
        # Check daily signal limit
        if strategy.max_signals_per_day:
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            today_signals = sum(
                1 for t in self.execution_history.get(strategy.strategy_id, [])
                if t >= today_start
            )
            if today_signals >= strategy.max_signals_per_day:
                return False
        
        # Check allowed hours
        if strategy.allowed_hours:
            start_hour, end_hour = strategy.allowed_hours
            if not (start_hour <= current_time.hour < end_hour):
                return False
        
        return True
    
    def _calculate_features(self, aligned_data: Dict[str, Event], sync_time: datetime) -> Dict[str, Any]:
        """Calculate features with caching."""
        # Create cache key
        cache_key = self._create_cache_key(aligned_data, sync_time)
        
        # Check cache
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Calculate features
        features = {}
        
        if self.feature_calculator:
            # Calculate per-symbol features
            for key, event in aligned_data.items():
                symbol = key.split('_')[0]
                if hasattr(self.feature_calculator, 'calculate_features'):
                    symbol_features = self.feature_calculator.calculate_features(event.payload)
                    features[symbol] = symbol_features
            
            # Calculate cross-symbol features
            if hasattr(self.feature_calculator, 'calculate_cross_features'):
                cross_features = self.feature_calculator.calculate_cross_features(aligned_data)
                features['_cross'] = cross_features
        
        # Cache results
        self.feature_cache[cache_key] = features
        
        # Maintain cache size
        if len(self.feature_cache) > self.feature_cache_size:
            # Remove oldest entry
            oldest = min(self.feature_cache.keys())
            del self.feature_cache[oldest]
        
        return features
    
    def _get_classification(self, features: Dict[str, Any], 
                           aligned_data: Dict[str, Event],
                           sync_time: datetime) -> Any:
        """Get market classification with caching."""
        # Create cache key
        cache_key = f"clf_{sync_time.isoformat()}"
        
        # Check cache
        if cache_key in self.classifier_cache:
            return self.classifier_cache[cache_key]
        
        # Get classification
        classification = None
        if hasattr(self.classifier, 'classify'):
            classification = self.classifier.classify(features, aligned_data)
        
        # Cache result
        self.classifier_cache[cache_key] = classification
        
        # Maintain cache size
        if len(self.classifier_cache) > 100:  # Smaller cache for classifications
            oldest = min(self.classifier_cache.keys())
            del self.classifier_cache[oldest]
        
        return classification
    
    def _process_signal(self,
                       signal: Dict[str, Any],
                       strategy: StrategySpecification,
                       features: Dict[str, Any],
                       classification: Optional[Any],
                       aligned_data: Dict[str, Event],
                       sync_time: datetime) -> None:
        """Process and publish enriched signal."""
        # Enrich signal
        enriched_signal = {
            **signal,
            'strategy_id': strategy.strategy_id,
            'sync_time': sync_time.isoformat(),
            'classification': str(classification) if classification else None,
            'bar_data': self._extract_bar_data(aligned_data),
            'features': features.get(signal.get('symbol'), {}),
            'parameters': strategy.parameters
        }
        
        # Create signal event
        signal_event = create_signal_event(
            symbol=signal['symbol'],
            direction=signal['direction'],
            strength=signal.get('strength', 1.0),
            strategy_id=strategy.strategy_id,
            source_id=self.container.container_id
        )
        
        # Update payload with enriched data
        signal_event.payload.update(enriched_signal)
        
        # Publish to parent
        self.container.publish_event(signal_event, target_scope="parent")
        
        logger.debug(f"Published signal from {strategy.strategy_id} for {signal['symbol']} at {sync_time}")
    
    def _group_strategies(self) -> None:
        """Group strategies by data requirements."""
        self.strategy_groups.clear()
        for strategy in self.strategies:
            key = strategy.data_requirement.to_key()
            self.strategy_groups[key].append(strategy)
    
    def _has_required_data(self, requirement: DataRequirement, aligned_data: Dict[str, Event]) -> bool:
        """Check if aligned data satisfies requirement."""
        if requirement.alignment_mode == AlignmentMode.ALL:
            # Need all required keys
            for symbol, timeframe in requirement.required_keys:
                if f"{symbol}_{timeframe}" not in aligned_data:
                    return False
            return True
        elif requirement.alignment_mode == AlignmentMode.ANY:
            # Need at least one
            for symbol, timeframe in requirement.required_keys:
                if f"{symbol}_{timeframe}" in aligned_data:
                    return True
            return False
        else:  # BEST_EFFORT or ROLLING
            # Always process what we have
            return len(aligned_data) > 0
    
    def _prepare_strategy_features(self, strategy: StrategySpecification, 
                                  features: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare features specific to strategy needs."""
        # Could filter or transform features based on strategy requirements
        return features
    
    def _check_classifier_gate(self, required_classifier: str, classification: Any) -> bool:
        """Check if classifier allows strategy execution."""
        # This is flexible - could check classifier ID, state, confidence, etc.
        if hasattr(classification, 'allows_trading'):
            return classification.allows_trading
        return True
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal has required fields."""
        required = ['symbol', 'direction']
        return all(field in signal for field in required)
    
    def _extract_bar_data(self, aligned_data: Dict[str, Event]) -> Dict[str, Any]:
        """Extract OHLCV data from aligned bars."""
        bar_data = {}
        for key, event in aligned_data.items():
            bar = event.payload.get('bar')
            if bar:
                bar_data[key] = {
                    'open': bar.open if hasattr(bar, 'open') else bar.get('open'),
                    'high': bar.high if hasattr(bar, 'high') else bar.get('high'),
                    'low': bar.low if hasattr(bar, 'low') else bar.get('low'),
                    'close': bar.close if hasattr(bar, 'close') else bar.get('close'),
                    'volume': bar.volume if hasattr(bar, 'volume') else bar.get('volume'),
                    'timestamp': event.payload.get('bar_close_time')
                }
        return bar_data
    
    def _create_cache_key(self, aligned_data: Dict[str, Event], sync_time: datetime) -> str:
        """Create cache key for feature storage."""
        # Use bar timestamps for cache key
        bar_times = []
        for key in sorted(aligned_data.keys()):
            event = aligned_data[key]
            bar_time = event.payload.get('bar_close_time', sync_time)
            if isinstance(bar_time, datetime):
                bar_time = bar_time.isoformat()
            bar_times.append(f"{key}:{bar_time}")
        
        return "|".join(bar_times)


def setup_synchronized_feature_container(container: 'Container', config: Dict[str, Any]) -> None:
    """
    Setup a feature container with full synchronization and orchestration.
    
    Example config:
    {
        'buffer_size': 1000,
        'strategies': [
            {
                'id': 'multi_asset_momentum',
                'function': momentum_strategy_func,
                'symbols': ['SPY', 'QQQ', 'IWM'],
                'timeframes': ['1m', '5m'],
                'min_history': 20,
                'classifier_id': 'trend',
                'parameters': {'lookback': 20, 'threshold': 0.02},
                'min_time_between_signals': timedelta(minutes=5),
                'max_signals_per_day': 10
            },
            {
                'id': 'pairs_arbitrage',
                'function': pairs_strategy_func,
                'symbols': ['AAPL', 'MSFT'],
                'timeframes': ['1m'],
                'alignment_mode': 'all',
                'parameters': {'spread_threshold': 2.0}
            }
        ]
    }
    """
    # Create time alignment buffer
    time_buffer = TimeAlignmentBuffer(
        buffer_size=config.get('buffer_size', 1000),
        default_alignment=TimeframeAlignment[config.get('alignment', 'NEAREST')],
        emit_partial=config.get('emit_partial', False)
    )
    container.add_component('time_buffer', time_buffer)
    
    # Create strategy specifications
    strategy_specs = []
    for strat_config in config.get('strategies', []):
        # Create data requirement
        data_req = DataRequirement(
            symbols=strat_config['symbols'],
            timeframes=strat_config['timeframes'],
            min_history=strat_config.get('min_history', 1),
            alignment_mode=AlignmentMode[strat_config.get('alignment_mode', 'ALL').upper()],
            timeout_ms=strat_config.get('timeout_ms')
        )
        
        # Create strategy specification
        spec = StrategySpecification(
            strategy_id=strat_config['id'],
            strategy_function=strat_config['function'],
            data_requirement=data_req,
            classifier_id=strat_config.get('classifier_id'),
            parameters=strat_config.get('parameters', {}),
            min_time_between_signals=strat_config.get('min_time_between_signals'),
            max_signals_per_day=strat_config.get('max_signals_per_day'),
            allowed_hours=strat_config.get('allowed_hours')
        )
        strategy_specs.append(spec)
    
    # Create orchestrator
    orchestrator = StrategyOrchestrator(
        strategies=strategy_specs,
        feature_cache_size=config.get('feature_cache_size', 1000),
        enable_execution_tracking=config.get('enable_execution_tracking', True)
    )
    container.add_component('strategy_orchestrator', orchestrator)
    
    logger.info(f"Setup synchronized feature container with {len(strategy_specs)} strategies")


# Utility functions for common patterns

def create_symbol_group_requirement(symbols: List[str], timeframe: str = '1m') -> DataRequirement:
    """Create requirement for a group of symbols at same timeframe."""
    return DataRequirement(
        symbols=symbols,
        timeframes=[timeframe],
        alignment_mode=AlignmentMode.ALL
    )


def create_multi_timeframe_requirement(symbol: str, timeframes: List[str]) -> DataRequirement:
    """Create requirement for multiple timeframes of same symbol."""
    return DataRequirement(
        symbols=[symbol],
        timeframes=timeframes,
        alignment_mode=AlignmentMode.ALL,
        timeframe_alignment=TimeframeAlignment.FORWARD_FILL
    )


def create_pairs_requirement(symbol1: str, symbol2: str, timeframe: str = '1m') -> DataRequirement:
    """Create requirement for pairs trading."""
    return DataRequirement(
        symbols=[symbol1, symbol2],
        timeframes=[timeframe],
        alignment_mode=AlignmentMode.ALL,
        timeframe_alignment=TimeframeAlignment.STRICT
    )