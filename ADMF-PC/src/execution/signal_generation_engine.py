"""
Signal Generation Container for pure signal analysis.

Implements Pattern #3 from BACKTEST.MD - signal generation without execution.
"""
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import uuid

from ..core.containers import UniversalScopedContainer
from ..core.containers.factory import ContainerFactory
from ..core.components import ComponentSpec
from .analysis.signal_analysis import SignalAnalysisEngine, AnalysisType
from ..core.events import Event, EventType
from ..data.models import MarketData
from ..strategy.protocols import Strategy
from ..risk.protocols import Signal


class SignalGenerationContainer(UniversalScopedContainer):
    """
    Container for signal generation and analysis only - no execution.
    
    This container:
    - Streams historical data
    - Computes indicators
    - Runs classifiers for regime detection
    - Generates signals from strategies
    - Analyzes signal quality (MAE/MFE, forward returns)
    - NO risk assessment, NO order generation, NO execution
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        analysis_config: Optional[Dict[str, Any]] = None,
        shared_services: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize signal generation container.
        
        Args:
            container_id: Unique container ID
            analysis_config: Configuration for signal analysis
            shared_services: Shared read-only services
        """
        # Generate structured container ID
        if container_id is None:
            container_id = f"signal_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
        super().__init__(
            container_id=container_id,
            container_type="signal_generation",
            shared_services=shared_services
        )
        
        self.analysis_config = analysis_config or {}
        
        # Core components
        self.data_streamer = None
        self.indicator_hub = None
        self.classifiers: Dict[str, Any] = {}
        self.strategies: List[Strategy] = []
        self.analysis_engine: Optional[SignalAnalysisEngine] = None
        
        # Signal tracking
        self._signal_count = 0
        self._start_time = None
        self._end_time = None
        
    def setup_components(self, config: Dict[str, Any]) -> None:
        """Set up container components based on configuration."""
        # Create data streamer
        if 'data_config' in config:
            self._create_data_streamer(config['data_config'])
            
        # Create indicator hub
        if 'indicator_config' in config:
            self._create_indicator_hub(config['indicator_config'])
            
        # Create classifiers
        if 'classifiers' in config:
            for classifier_config in config['classifiers']:
                self._create_classifier(classifier_config)
                
        # Create strategies
        if 'strategies' in config:
            for strategy_config in config['strategies']:
                self._create_strategy(strategy_config)
                
        # Create analysis engine
        self._create_analysis_engine(config.get('analysis_config', {}))
        
    def _create_data_streamer(self, data_config: Dict[str, Any]) -> None:
        """Create data streaming component."""
        spec = ComponentSpec(
            name="data_streamer",
            class_name=data_config.get('class', 'HistoricalDataStreamer'),
            parameters=data_config.get('parameters', {}),
            capabilities=['event_publisher']
        )
        self.data_streamer = self.create_component(spec)
        
    def _create_indicator_hub(self, indicator_config: Dict[str, Any]) -> None:
        """Create centralized indicator hub."""
        spec = ComponentSpec(
            name="indicator_hub",
            class_name="IndicatorHub",
            parameters={
                'indicators': indicator_config.get('indicators', []),
                'cache_enabled': indicator_config.get('cache_enabled', True)
            },
            capabilities=['event_publisher', 'event_subscriber']
        )
        self.indicator_hub = self.create_component(spec)
        
        # Subscribe to market data events
        self.event_bus.subscribe(EventType.BAR, self.indicator_hub.process_market_data)
        self.event_bus.subscribe(EventType.TICK, self.indicator_hub.process_market_data)
        
    def _create_classifier(self, classifier_config: Dict[str, Any]) -> None:
        """Create a classifier container."""
        classifier_type = classifier_config.get('type', 'hmm')
        classifier_id = f"{self.container_id}_{classifier_type}"
        
        # Create sub-container for classifier
        classifier_container = self.create_subcontainer(
            container_id=classifier_id,
            container_type=f"classifier_{classifier_type}"
        )
        
        # Create classifier component
        spec = ComponentSpec(
            name=f"classifier_{classifier_type}",
            class_name=classifier_config.get('class', f'{classifier_type.upper()}Classifier'),
            parameters=classifier_config.get('parameters', {}),
            capabilities=['event_publisher', 'event_subscriber']
        )
        
        classifier = classifier_container.create_component(spec)
        self.classifiers[classifier_type] = classifier
        
        # Subscribe to indicator events
        if self.indicator_hub:
            classifier_container.event_bus.subscribe(
                EventType.INDICATOR,
                classifier.process_indicator_event
            )
            
    def _create_strategy(self, strategy_config: Dict[str, Any]) -> None:
        """Create a strategy component."""
        spec = ComponentSpec(
            name=strategy_config.get('name', f"strategy_{len(self.strategies)}"),
            class_name=strategy_config.get('class'),
            parameters=strategy_config.get('parameters', {}),
            capabilities=['event_publisher']
        )
        
        strategy = self.create_component(spec)
        self.strategies.append(strategy)
        
        # Subscribe strategy to market and indicator events
        self.event_bus.subscribe(EventType.BAR, strategy.on_market_data)
        self.event_bus.subscribe(EventType.INDICATOR, strategy.on_indicator_update)
        
    def _create_analysis_engine(self, analysis_config: Dict[str, Any]) -> None:
        """Create the signal analysis engine."""
        self.analysis_engine = SignalAnalysisEngine(
            lookback_bars=analysis_config.get('lookback_bars', 20),
            forward_bars=analysis_config.get('forward_bars', [1, 5, 10, 20]),
            analysis_types=analysis_config.get(
                'analysis_types',
                [AnalysisType.MAE_MFE, AnalysisType.FORWARD_RETURNS, AnalysisType.SIGNAL_QUALITY]
            )
        )
        
        # Give it access to event bus
        self.analysis_engine.set_event_bus(self.event_bus)
        
        # Subscribe to signal events
        self.event_bus.subscribe(EventType.SIGNAL, self._handle_signal_event)
        
    def _handle_signal_event(self, event: Event) -> None:
        """Handle signal events for analysis."""
        if self.analysis_engine and isinstance(event.payload, dict):
            signal_data = event.payload.get('signal')
            market_data = event.payload.get('market_data', {})
            
            if signal_data:
                # Convert to Signal object if needed
                if isinstance(signal_data, dict):
                    signal = Signal(**signal_data)
                else:
                    signal = signal_data
                    
                self.analysis_engine.capture_signal(signal, market_data)
                self._signal_count += 1
                
    async def run_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Run signal generation and analysis.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            symbols: List of symbols to analyze
            
        Returns:
            Analysis results
        """
        self._start_time = datetime.now()
        
        # Initialize components
        await self.initialize()
        await self.start()
        
        # Configure data streaming
        if self.data_streamer:
            await self.data_streamer.configure({
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols
            })
            
        # Main analysis loop
        self.logger.info(
            f"Starting signal generation analysis",
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        # Stream data and generate signals
        async for timestamp, market_data in self.data_streamer.stream():
            # Update analysis engine with market data
            if self.analysis_engine:
                self.analysis_engine.process_market_data(timestamp, market_data)
                
            # Publish market data event
            self.event_bus.publish(Event(
                event_type=EventType.BAR,
                payload={
                    'timestamp': timestamp,
                    'market_data': market_data
                },
                source_id=self.container_id
            ))
            
            # Generate signals from each strategy
            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(market_data, timestamp)
                    
                    # Publish signal events
                    for signal in signals:
                        self.event_bus.publish(Event(
                            event_type=EventType.SIGNAL,
                            payload={
                                'signal': signal,
                                'market_data': market_data,
                                'timestamp': timestamp
                            },
                            source_id=strategy.name
                        ))
                except Exception as e:
                    self.logger.error(
                        f"Error generating signals from {strategy.name}: {e}"
                    )
                    
        self._end_time = datetime.now()
        
        # Get analysis results
        if self.analysis_engine:
            results = self.analysis_engine.get_analysis_results()
        else:
            results = {}
            
        # Add metadata
        results['metadata'] = {
            'container_id': self.container_id,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'symbols': symbols,
            'total_signals': self._signal_count,
            'analysis_duration': (self._end_time - self._start_time).total_seconds(),
            'strategies': [s.name for s in self.strategies],
            'classifiers': list(self.classifiers.keys())
        }
        
        # Clean up
        await self.stop()
        
        return results
        
    def export_signals(self, filepath: str) -> None:
        """Export analyzed signals to file."""
        if self.analysis_engine:
            self.analysis_engine.export_signals(filepath)
            
    def get_optimal_stops(self) -> Dict[str, float]:
        """Get optimal stop loss and take profit levels from MAE/MFE analysis."""
        if not self.analysis_engine:
            return {}
            
        mae_mfe_summary = self.analysis_engine._get_mae_mfe_summary()
        
        return {
            'optimal_stop_loss': mae_mfe_summary.get('optimal_stop_loss', 0.02),
            'optimal_take_profit': mae_mfe_summary.get('optimal_take_profit', 0.05)
        }
        
    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by signal quality metrics."""
        if not self.analysis_engine:
            return []
            
        metrics = self.analysis_engine.calculate_strategy_metrics()
        
        # Rank by expectancy
        rankings = []
        for strategy_id, strategy_metrics in metrics.items():
            rankings.append({
                'strategy_id': strategy_id,
                'expectancy': strategy_metrics.get('expectancy', 0),
                'win_rate': strategy_metrics.get('win_rate', 0),
                'total_signals': strategy_metrics.get('total_signals', 0),
                'sharpe_1bar': strategy_metrics.get('sharpe_1bar', 0)
            })
            
        rankings.sort(key=lambda x: x['expectancy'], reverse=True)
        
        return rankings


class SignalGenerationContainerFactory:
    """Factory for creating signal generation containers."""
    
    @staticmethod
    def create_instance(config: Dict[str, Any]) -> SignalGenerationContainer:
        """
        Create a signal generation container instance.
        
        Args:
            config: Container configuration including:
                - data_config: Data streaming configuration
                - indicator_config: Indicator hub configuration
                - classifiers: List of classifier configurations
                - strategies: List of strategy configurations
                - analysis_config: Signal analysis configuration
                
        Returns:
            Configured SignalGenerationContainer
        """
        # Extract container ID if provided
        container_id = config.get('container_id')
        
        # Create container
        container = SignalGenerationContainer(
            container_id=container_id,
            analysis_config=config.get('analysis_config', {}),
            shared_services=config.get('shared_services', {})
        )
        
        # Set up components
        container.setup_components(config)
        
        # Wire event flows
        container._wire_signal_generation_flows()
        
        return container
        
    @staticmethod
    def create_from_workflow(
        workflow_config: Dict[str, Any],
        phase: str = "signal_analysis"
    ) -> SignalGenerationContainer:
        """
        Create container from workflow configuration.
        
        Args:
            workflow_config: Workflow configuration
            phase: Current workflow phase
            
        Returns:
            Configured SignalGenerationContainer
        """
        # Extract relevant config for signal generation
        config = {
            'container_id': f"signal_gen_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'data_config': workflow_config.get('data_config', {}),
            'indicator_config': workflow_config.get('shared_indicators', {}),
            'classifiers': workflow_config.get('classifiers', []),
            'strategies': workflow_config.get('strategies', []),
            'analysis_config': workflow_config.get('signal_analysis', {
                'lookback_bars': 20,
                'forward_bars': [1, 5, 10, 20],
                'analysis_types': ['mae_mfe', 'forward_returns', 'signal_quality', 'correlation']
            }),
            'shared_services': workflow_config.get('shared_services', {})
        }
        
        return SignalGenerationContainerFactory.create_instance(config)