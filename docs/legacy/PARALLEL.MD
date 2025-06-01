"""
Canonical Parallel Backtester Implementation

This is the definitive backtester that implements the architectural vision:
- Single data iteration with parallel component processing
- Component subscription model for efficient indicator distribution
- Configurable parallelization limits to avoid system overload
- Results streaming to prevent memory issues
- Clean separation from YAML config (handled by Coordinator)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import json
import gzip
from collections import defaultdict, deque
import uuid
import hashlib

from ..core.containers import UniversalScopedContainer
from ..core.events import Event, EventType, EventBus
from ..data.protocols import DataLoader
from ..strategy.protocols import Strategy
from ..risk.protocols import Signal, PortfolioStateProtocol
from ..strategy.components import IndicatorHub

logger = logging.getLogger(__name__)


@dataclass
class ParallelizationLimits:
    """Configuration for parallelization limits."""
    max_parallel_classifiers: int = 10
    max_parallel_risk_containers: int = 20
    max_parallel_strategies: int = 50
    max_total_components: int = 100
    max_memory_usage_mb: int = 2048
    results_buffer_size: int = 1000


@dataclass
class ComponentSubscription:
    """Represents a component's subscription to specific indicators."""
    component_id: str
    component_type: str  # 'classifier', 'strategy', etc.
    required_indicators: Set[str]
    callback: Callable[[str, Any, datetime], None]


@dataclass
class ProcessingPath:
    """Represents a unique processing path through the system."""
    path_id: str
    classifier_type: str
    classifier_params: Dict[str, Any]
    risk_profile: str
    risk_params: Dict[str, Any]
    strategy_type: str
    strategy_params: Dict[str, Any]
    symbols: List[str]
    
    # Component instances
    classifier: Optional[Any] = None
    risk_manager: Optional[Any] = None
    strategy: Optional[Strategy] = None
    
    # Results tracking
    signals_generated: int = 0
    orders_generated: int = 0
    current_equity: float = 0.0


class SubscriptionBasedIndicatorHub:
    """
    Enhanced IndicatorHub with component subscription model.
    
    Components subscribe to specific indicators they need, not all indicators.
    """
    
    def __init__(self, max_indicators: int = 100):
        self.max_indicators = max_indicators
        self.indicators: Dict[str, Any] = {}
        self.computed_values: Dict[str, Any] = {}
        self.subscriptions: Dict[str, List[ComponentSubscription]] = defaultdict(list)
        self.computation_cache: Dict[str, deque] = {}
        
        logger.info(f"IndicatorHub initialized with max {max_indicators} indicators")
    
    def register_indicator(self, name: str, indicator_config: Dict[str, Any]) -> None:
        """Register an indicator for computation."""
        if len(self.indicators) >= self.max_indicators:
            raise ValueError(f"Maximum indicators ({self.max_indicators}) exceeded")
            
        self.indicators[name] = indicator_config
        self.computation_cache[name] = deque(maxlen=1000)  # Keep recent values
        
        logger.debug(f"Registered indicator: {name}")
    
    def subscribe_component(
        self, 
        component_id: str,
        component_type: str,
        required_indicators: Set[str],
        callback: Callable[[str, Any, datetime], None]
    ) -> None:
        """Subscribe component to specific indicators."""
        subscription = ComponentSubscription(
            component_id=component_id,
            component_type=component_type,
            required_indicators=required_indicators,
            callback=callback
        )
        
        for indicator_name in required_indicators:
            if indicator_name not in self.indicators:
                logger.warning(f"Component {component_id} subscribing to unknown indicator: {indicator_name}")
                continue
                
            self.subscriptions[indicator_name].append(subscription)
            
        logger.debug(f"Component {component_id} subscribed to {len(required_indicators)} indicators")
    
    def process_market_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Process market data and notify subscribers."""
        
        # Compute ALL registered indicators (done once)
        computed_indicators = {}
        for indicator_name, indicator_config in self.indicators.items():
            try:
                # Compute indicator value
                value = self._compute_indicator(indicator_config, market_data, timestamp)
                computed_indicators[indicator_name] = value
                self.computed_values[indicator_name] = value
                
                # Cache for history
                self.computation_cache[indicator_name].append((timestamp, value))
                
            except Exception as e:
                logger.error(f"Error computing indicator {indicator_name}: {e}")
                continue
        
        # Notify ONLY subscribers of each indicator
        for indicator_name, value in computed_indicators.items():
            subscriptions = self.subscriptions.get(indicator_name, [])
            
            for subscription in subscriptions:
                try:
                    subscription.callback(indicator_name, value, timestamp)
                except Exception as e:
                    logger.error(f"Error notifying {subscription.component_id} of {indicator_name}: {e}")
    
    def _compute_indicator(self, config: Dict[str, Any], market_data: Dict[str, Any], timestamp: datetime) -> Any:
        """Compute individual indicator value."""
        indicator_type = config.get('type', 'unknown')
        params = config.get('params', {})
        
        # This would integrate with your existing indicator computation logic
        # For now, simplified implementation
        if indicator_type == 'SMA':
            period = params.get('period', 20)
            symbol = params.get('symbol', list(market_data.get('prices', {}).keys())[0])
            price = market_data.get('prices', {}).get(symbol, 100.0)
            
            # Get historical values from cache
            historical = list(self.computation_cache.get(config['name'], []))
            prices = [price] + [val for _, val in historical[-period+1:]]
            return sum(prices) / len(prices)
            
        elif indicator_type == 'RSI':
            # Simplified RSI calculation
            return 50.0  # Placeholder
            
        else:
            # Default/unknown indicator
            return 0.0
    
    def get_indicator_history(self, indicator_name: str, bars: int = 20) -> List[Tuple[datetime, Any]]:
        """Get recent history for an indicator."""
        cache = self.computation_cache.get(indicator_name, deque())
        return list(cache)[-bars:]


class StreamingResultsManager:
    """
    Manages streaming results to disk to prevent memory overflow.
    
    Results are written immediately as they're generated.
    Only summary metrics kept in memory.
    """
    
    def __init__(self, output_dir: Path, buffer_size: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        
        # In-memory summaries (minimal)
        self.path_summaries: Dict[str, Dict[str, Any]] = {}
        self.result_buffers: Dict[str, List[Dict[str, Any]]] = {}
        
        # File handles for streaming
        self.result_files: Dict[str, Any] = {}
        
        # Performance tracking
        self.total_results_written = 0
        self.paths_being_tracked = 0
        
        logger.info(f"Results manager initialized - output: {output_dir}, buffer: {buffer_size}")
    
    def stream_path_result(
        self, 
        path_id: str, 
        timestamp: datetime, 
        result: Dict[str, Any]
    ) -> None:
        """Stream result for a specific processing path."""
        
        # Update summary
        if path_id not in self.path_summaries:
            self.path_summaries[path_id] = {
                'first_timestamp': timestamp,
                'last_timestamp': timestamp,
                'total_signals': 0,
                'total_orders': 0,
                'current_equity': result.get('equity', 0.0),
                'max_equity': result.get('equity', 0.0),
                'min_equity': result.get('equity', 0.0),
                'path_info': result.get('path_info', {})
            }
            self.paths_being_tracked += 1
        
        summary = self.path_summaries[path_id]
        summary['last_timestamp'] = timestamp
        summary['total_signals'] += len(result.get('signals', []))
        summary['total_orders'] += len(result.get('orders', []))
        summary['current_equity'] = result.get('equity', summary['current_equity'])
        summary['max_equity'] = max(summary['max_equity'], summary['current_equity'])
        summary['min_equity'] = min(summary['min_equity'], summary['current_equity'])
        
        # Buffer result
        if path_id not in self.result_buffers:
            self.result_buffers[path_id] = []
        
        self.result_buffers[path_id].append({
            'timestamp': timestamp.isoformat(),
            'data': result
        })
        
        # Flush if buffer full
        if len(self.result_buffers[path_id]) >= self.buffer_size:
            self._flush_path_buffer(path_id)
    
    def _flush_path_buffer(self, path_id: str) -> None:
        """Flush buffered results for a path to disk."""
        if path_id not in self.result_buffers or not self.result_buffers[path_id]:
            return
        
        # Get or create file handle
        if path_id not in self.result_files:
            result_file = self.output_dir / f"{path_id}_results.jsonl.gz"
            self.result_files[path_id] = gzip.open(result_file, 'at')
        
        # Write buffered results
        file_handle = self.result_files[path_id]
        buffer = self.result_buffers[path_id]
        
        for result_entry in buffer:
            json.dump(result_entry, file_handle)
            file_handle.write('\n')
            self.total_results_written += 1
        
        file_handle.flush()
        
        # Clear buffer
        buffer.clear()
        
        logger.debug(f"Flushed {len(buffer)} results for path {path_id}")
    
    def finalize_results(self) -> Dict[str, Any]:
        """Finalize all results and close files."""
        
        # Flush all remaining buffers
        for path_id in list(self.result_buffers.keys()):
            self._flush_path_buffer(path_id)
        
        # Close all file handles
        for file_handle in self.result_files.values():
            file_handle.close()
        
        # Create results manifest
        manifest = {
            'output_directory': str(self.output_dir),
            'total_paths': self.paths_being_tracked,
            'total_results_written': self.total_results_written,
            'path_summaries': self.path_summaries,
            'result_files': {
                path_id: str(self.output_dir / f"{path_id}_results.jsonl.gz")
                for path_id in self.path_summaries.keys()
            }
        }
        
        # Save manifest
        manifest_file = self.output_dir / "results_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        logger.info(f"Results finalized - {self.paths_being_tracked} paths, {self.total_results_written} total results")
        
        return manifest


class ParallelBacktester:
    """
    Canonical parallel backtester implementation.
    
    This is the definitive backtester that implements the architectural vision.
    All other backtest implementations should be replaced with this one.
    """
    
    def __init__(
        self,
        processing_paths: List[ProcessingPath],
        parallelization_limits: ParallelizationLimits,
        output_dir: Path,
        enable_signal_capture: bool = True
    ):
        """
        Initialize parallel backtester.
        
        Args:
            processing_paths: All parameter combinations to test in parallel
            parallelization_limits: Limits to prevent system overload
            output_dir: Directory for streaming results
            enable_signal_capture: Whether to capture signals for Phase 3
        """
        self.processing_paths = processing_paths
        self.limits = parallelization_limits
        self.output_dir = Path(output_dir)
        self.enable_signal_capture = enable_signal_capture
        
        # Validate parallelization limits
        self._validate_parallelization_limits()
        
        # Core components
        self.indicator_hub = SubscriptionBasedIndicatorHub()
        self.results_manager = StreamingResultsManager(output_dir, parallelization_limits.results_buffer_size)
        self.event_bus = EventBus()
        
        # Processing state
        self.is_initialized = False
        self.is_running = False
        self.current_timestamp: Optional[datetime] = None
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.bars_processed = 0
        self.total_signals_generated = 0
        
        logger.info(
            f"ParallelBacktester initialized - {len(processing_paths)} paths, "
            f"limits: {parallelization_limits.max_total_components} components, "
            f"output: {output_dir}"
        )
    
    def _validate_parallelization_limits(self) -> None:
        """Validate that processing paths don't exceed limits."""
        total_paths = len(self.processing_paths)
        
        if total_paths > self.limits.max_total_components:
            raise ValueError(
                f"Total processing paths ({total_paths}) exceeds limit "
                f"({self.limits.max_total_components})"
            )
        
        # Count components by type
        classifiers = len(set(path.classifier_type for path in self.processing_paths))
        risk_containers = len(set(f"{path.classifier_type}_{path.risk_profile}" for path in self.processing_paths))
        
        if classifiers > self.limits.max_parallel_classifiers:
            raise ValueError(f"Too many parallel classifiers: {classifiers} > {self.limits.max_parallel_classifiers}")
        
        if risk_containers > self.limits.max_parallel_risk_containers:
            raise ValueError(f"Too many parallel risk containers: {risk_containers} > {self.limits.max_parallel_risk_containers}")
        
        logger.info(f"Parallelization validation passed - {total_paths} paths, {classifiers} classifiers, {risk_containers} risk containers")
    
    def setup_parallel_components(self) -> None:
        """Set up all parallel components and their subscriptions."""
        
        # Collect all required indicators
        all_required_indicators = set()
        for path in self.processing_paths:
            # Extract indicator requirements from strategy and classifier configs
            strategy_indicators = self._get_strategy_indicators(path.strategy_type, path.strategy_params)
            classifier_indicators = self._get_classifier_indicators(path.classifier_type, path.classifier_params)
            
            all_required_indicators.update(strategy_indicators)
            all_required_indicators.update(classifier_indicators)
        
        # Register all indicators with the hub
        for indicator_name in all_required_indicators:
            indicator_config = self._create_indicator_config(indicator_name)
            self.indicator_hub.register_indicator(indicator_name, indicator_config)
        
        # Create and subscribe all components
        for path in self.processing_paths:
            self._setup_processing_path(path)
        
        logger.info(f"Parallel components setup complete - {len(all_required_indicators)} indicators, {len(self.processing_paths)} paths")
    
    def _get_strategy_indicators(self, strategy_type: str, params: Dict[str, Any]) -> Set[str]:
        """Get indicator requirements for a strategy."""
        # This would integrate with your strategy registry
        # For now, basic mapping
        indicators = set()
        
        if strategy_type == 'MovingAverageCrossover':
            fast_period = params.get('fast_period', 10)
            slow_period = params.get('slow_period', 20)
            indicators.add(f'SMA_{fast_period}')
            indicators.add(f'SMA_{slow_period}')
            
        elif strategy_type == 'RSIMeanReversion':
            rsi_period = params.get('rsi_period', 14)
            indicators.add(f'RSI_{rsi_period}')
            
        elif strategy_type == 'MomentumStrategy':
            lookback = params.get('lookback_period', 20)
            indicators.add(f'ROC_{lookback}')
            indicators.add('Volume')
            
        return indicators
    
    def _get_classifier_indicators(self, classifier_type: str, params: Dict[str, Any]) -> Set[str]:
        """Get indicator requirements for a classifier."""
        indicators = set()
        
        if classifier_type == 'HMM':
            indicators.add('RSI_14')
            indicators.add('MACD')
            indicators.add('ATR_14')
            
        elif classifier_type == 'PatternBased':
            indicators.add('Volume')
            indicators.add('ATR_20')
            indicators.add('BollingerBands_20')
            
        return indicators
    
    def _create_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """Create configuration for an indicator."""
        # Parse indicator name to extract type and parameters
        if '_' in indicator_name:
            indicator_type, param_str = indicator_name.split('_', 1)
            try:
                period = int(param_str)
                return {
                    'name': indicator_name,
                    'type': indicator_type,
                    'params': {'period': period}
                }
            except ValueError:
                pass
        
        # Default configuration
        return {
            'name': indicator_name,
            'type': indicator_name,
            'params': {}
        }
    
    def _setup_processing_path(self, path: ProcessingPath) -> None:
        """Set up components for a single processing path."""
        
        # Create classifier component
        path.classifier = self._create_classifier(path.classifier_type, path.classifier_params)
        
        # Create risk manager component
        path.risk_manager = self._create_risk_manager(path.risk_profile, path.risk_params)
        
        # Create strategy component
        path.strategy = self._create_strategy(path.strategy_type, path.strategy_params, path.symbols)
        
        # Subscribe components to required indicators
        self._subscribe_path_components(path)
        
        logger.debug(f"Processing path setup complete: {path.path_id}")
    
    def _create_classifier(self, classifier_type: str, params: Dict[str, Any]) -> Any:
        """Create classifier component."""
        # This would integrate with your classifier factory
        return {
            'type': classifier_type,
            'params': params,
            'state': {},
            'process_indicator': self._create_classifier_processor(classifier_type, params)
        }
    
    def _create_risk_manager(self, risk_profile: str, params: Dict[str, Any]) -> Any:
        """Create risk manager component."""
        # This would integrate with your risk manager factory
        return {
            'profile': risk_profile,
            'params': params,
            'state': {'cash': 100000.0, 'positions': {}},
            'process_signal': self._create_risk_processor(risk_profile, params)
        }
    
    def _create_strategy(self, strategy_type: str, params: Dict[str, Any], symbols: List[str]) -> Any:
        """Create strategy component."""
        # This would integrate with your strategy factory
        return {
            'type': strategy_type,
            'params': params,
            'symbols': symbols,
            'state': {},
            'generate_signals': self._create_strategy_processor(strategy_type, params, symbols)
        }
    
    def _subscribe_path_components(self, path: ProcessingPath) -> None:
        """Subscribe path components to required indicators."""
        
        # Subscribe strategy
        strategy_indicators = self._get_strategy_indicators(path.strategy_type, path.strategy_params)
        self.indicator_hub.subscribe_component(
            component_id=f"{path.path_id}_strategy",
            component_type="strategy",
            required_indicators=strategy_indicators,
            callback=lambda ind, val, ts: self._handle_strategy_indicator(path, ind, val, ts)
        )
        
        # Subscribe classifier
        classifier_indicators = self._get_classifier_indicators(path.classifier_type, path.classifier_params)
        self.indicator_hub.subscribe_component(
            component_id=f"{path.path_id}_classifier",
            component_type="classifier",
            required_indicators=classifier_indicators,
            callback=lambda ind, val, ts: self._handle_classifier_indicator(path, ind, val, ts)
        )
    
    def _handle_strategy_indicator(self, path: ProcessingPath, indicator: str, value: Any, timestamp: datetime) -> None:
        """Handle indicator update for strategy."""
        if path.strategy and 'indicator_values' not in path.strategy['state']:
            path.strategy['state']['indicator_values'] = {}
        
        path.strategy['state']['indicator_values'][indicator] = value
    
    def _handle_classifier_indicator(self, path: ProcessingPath, indicator: str, value: Any, timestamp: datetime) -> None:
        """Handle indicator update for classifier."""
        if path.classifier and 'indicator_values' not in path.classifier['state']:
            path.classifier['state']['indicator_values'] = {}
        
        path.classifier['state']['indicator_values'][indicator] = value
    
    def _create_classifier_processor(self, classifier_type: str, params: Dict[str, Any]) -> Callable:
        """Create classifier processing function."""
        def process_indicators(indicator_values: Dict[str, Any]) -> str:
            # Simplified classifier logic
            if classifier_type == 'HMM':
                rsi = indicator_values.get('RSI_14', 50)
                if rsi > 70:
                    return 'overbought'
                elif rsi < 30:
                    return 'oversold'
                else:
                    return 'neutral'
            else:
                return 'unknown'
        
        return process_indicators
    
    def _create_risk_processor(self, risk_profile: str, params: Dict[str, Any]) -> Callable:
        """Create risk processing function."""
        def process_signal(signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Simplified risk management
            max_position = params.get('max_position_size', 0.02)
            if signal['strength'] > 0.5:  # Only process strong signals
                return {
                    'symbol': signal['symbol'],
                    'side': signal['direction'],
                    'quantity': max_position * 1000,  # Simplified position sizing
                    'order_type': 'market'
                }
            return None
        
        return process_signal
    
    def _create_strategy_processor(self, strategy_type: str, params: Dict[str, Any], symbols: List[str]) -> Callable:
        """Create strategy processing function."""
        def generate_signals(indicator_values: Dict[str, Any], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
            signals = []
            
            if strategy_type == 'MovingAverageCrossover':
                fast_period = params.get('fast_period', 10)
                slow_period = params.get('slow_period', 20)
                
                fast_ma = indicator_values.get(f'SMA_{fast_period}', 0)
                slow_ma = indicator_values.get(f'SMA_{slow_period}', 0)
                
                for symbol in symbols:
                    if fast_ma > slow_ma:
                        signals.append({
                            'symbol': symbol,
                            'direction': 'buy',
                            'strength': 0.8,
                            'timestamp': self.current_timestamp,
                            'strategy': strategy_type
                        })
                    elif fast_ma < slow_ma:
                        signals.append({
                            'symbol': symbol,
                            'direction': 'sell',
                            'strength': 0.6,
                            'timestamp': self.current_timestamp,
                            'strategy': strategy_type
                        })
            
            return signals
        
        return generate_signals
    
    async def run_parallel_backtest(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Run the parallel backtest - single data pass, all paths process simultaneously.
        
        Args:
            data_loader: Data source
            
        Returns:
            Backtest results
        """
        self.start_time = datetime.now()
        self.is_running = True
        
        logger.info("Starting parallel backtest execution")
        
        # Setup all components
        if not self.is_initialized:
            self.setup_parallel_components()
            self.is_initialized = True
        
        # Single data iteration - ALL paths process simultaneously
        async for timestamp, market_data in self._iterate_data(data_loader):
            self.current_timestamp = timestamp
            self.bars_processed += 1
            
            # 1. Process market data through IndicatorHub (compute ALL indicators once)
            self.indicator_hub.process_market_data(timestamp, market_data)
            
            # 2. Process all paths in parallel (they receive only subscribed indicators)
            path_results = await self._process_all_paths_parallel(timestamp, market_data)
            
            # 3. Stream results immediately to prevent memory issues
            for path_id, result in path_results.items():
                self.results_manager.stream_path_result(path_id, timestamp, result)
            
            # 4. Progress logging
            if self.bars_processed % 1000 == 0:
                logger.info(f"Processed {self.bars_processed} bars, {len(path_results)} active paths")
        
        # Finalize results
        final_results = self.results_manager.finalize_results()
        
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(
            f"Parallel backtest complete - {self.bars_processed} bars, "
            f"{len(self.processing_paths)} paths, {execution_time:.2f}s"
        )
        
        return {
            'execution_time_seconds': execution_time,
            'bars_processed': self.bars_processed,
            'total_paths': len(self.processing_paths),
            'total_signals_generated': self.total_signals_generated,
            'results_manifest': final_results
        }
    
    async def _iterate_data(self, data_loader: DataLoader):
        """Iterate through market data."""
        # This would integrate with your data loading system
        # For now, a simplified interface
        async for timestamp, market_data in data_loader.stream_data():
            yield timestamp, market_data
    
    async def _process_all_paths_parallel(
        self, 
        timestamp: datetime, 
        market_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Process all paths in parallel for current timestamp."""
        
        path_results = {}
        
        # Create tasks for all paths
        tasks = []
        for path in self.processing_paths:
            task = asyncio.create_task(
                self._process_single_path(path, timestamp, market_data)
            )
            tasks.append((path.path_id, task))
        
        # Execute all tasks concurrently
        for path_id, task in tasks:
            try:
                result = await task
                if result:
                    path_results[path_id] = result
            except Exception as e:
                logger.error(f"Error processing path {path_id}: {e}")
        
        return path_results
    
    async def _process_single_path(
        self, 
        path: ProcessingPath, 
        timestamp: datetime, 
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single path for current timestamp."""
        
        try:
            # 1. Get classifier state (regime detection)
            classifier_state = path.classifier['process_indicator'](
                path.classifier['state'].get('indicator_values', {})
            )
            
            # 2. Generate signals from strategy
            signals = path.strategy['generate_signals'](
                path.strategy['state'].get('indicator_values', {}),
                market_data
            )
            
            path.signals_generated += len(signals)
            self.total_signals_generated += len(signals)
            
            # 3. Process signals through risk management
            orders = []
            for signal in signals:
                order = path.risk_manager['process_signal'](signal, market_data)
                if order:
                    orders.append(order)
            
            path.orders_generated += len(orders)
            
            # 4. Update equity (simplified)
            path.current_equity = path.risk_manager['state']['cash']
            
            # 5. Create result for this path
            result = {
                'path_info': {
                    'path_id': path.path_id,
                    'classifier_type': path.classifier_type,
                    'risk_profile': path.risk_profile,
                    'strategy_type': path.strategy_type,
                    'symbols': path.symbols
                },
                'classifier_state': classifier_state,
                'signals': signals if self.enable_signal_capture else [],
                'orders': orders,
                'equity': path.current_equity,
                'performance': {
                    'total_signals': path.signals_generated,
                    'total_orders': path.orders_generated
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in path {path.path_id}: {e}")
            return None


def create_processing_paths_from_parameter_list(
    parameter_combinations: List[Dict[str, Any]]
) -> List[ProcessingPath]:
    """
    Create ProcessingPath objects from parameter combinations.
    
    This is where the Coordinator's parameter expansion gets converted
    to the Backtester's internal representation.
    """
    processing_paths = []
    
    for i, combo in enumerate(parameter_combinations):
        # Generate unique path ID
        path_id = _generate_path_id(combo, i)
        
        path = ProcessingPath(
            path_id=path_id,
            classifier_type=combo['classifier'],
            classifier_params=combo.get('classifier_params', {}),
            risk_profile=combo['risk_profile'],
            risk_params=combo.get('risk_params', {}),
            strategy_type=combo['strategy'],
            strategy_params=combo.get('strategy_params', {}),
            symbols=combo.get('symbols', ['SPY'])
        )
        
        processing_paths.append(path)
    
    return processing_paths


def _generate_path_id(combo: Dict[str, Any], index: int) -> str:
    """Generate unique path ID from combination parameters."""
    # Create deterministic ID
    combo_str = json.dumps({
        'classifier': combo['classifier'],
        'risk_profile': combo['risk_profile'],
        'strategy': combo['strategy'],
        'strategy_params': combo.get('strategy_params', {})
    }, sort_keys=True)
    
    combo_hash = hashlib.md5(combo_str.encode()).hexdigest()[:8]
    
    return f"{combo['classifier']}_{combo['risk_profile']}_{combo['strategy']}_{combo_hash}_{index}"


# Factory function for Coordinator integration
def create_parallel_backtester(
    parameter_combinations: List[Dict[str, Any]],
    parallelization_limits: Optional[Dict[str, int]] = None,
    output_dir: str = "./results",
    enable_signal_capture: bool = True
) -> ParallelBacktester:
    """
    Factory function to create ParallelBacktester from parameter combinations.
    
    This is the interface the Coordinator should use.
    
    Args:
        parameter_combinations: List of parameter combinations from Optimizer
        parallelization_limits: Limits to prevent system overload
        output_dir: Directory for results
        enable_signal_capture: Whether to capture signals for Phase 3
        
    Returns:
        Configured ParallelBacktester
    """
    # Convert parameter combinations to processing paths
    processing_paths = create_processing_paths_from_parameter_list(parameter_combinations)
    
    # Create parallelization limits
    if parallelization_limits is None:
        parallelization_limits = {}
    
    limits = ParallelizationLimits(
        max_parallel_classifiers=parallelization_limits.get('max_parallel_classifiers', 10),
        max_parallel_risk_containers=parallelization_limits.get('max_parallel_risk_containers', 20),
        max_parallel_strategies=parallelization_limits.get('max_parallel_strategies', 50),
        max_total_components=parallelization_limits.get('max_total_components', 100),
        max_memory_usage_mb=parallelization_limits.get('max_memory_usage_mb', 2048),
        results_buffer_size=parallelization_limits.get('results_buffer_size', 1000)
    )
    
    return ParallelBacktester(
        processing_paths=processing_paths,
        parallelization_limits=limits,
        output_dir=Path(output_dir),
        enable_signal_capture=enable_signal_capture
    )