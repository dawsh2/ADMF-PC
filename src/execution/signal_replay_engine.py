"""
Signal Replay Engine for ensemble optimization.

Implements Pattern #2 from BACKTEST.MD - replaying captured signals
with different weights for fast ensemble optimization.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import uuid
from pathlib import Path

from ..core.containers import UniversalScopedContainer
from ..core.components import ComponentSpec
from ..core.events import Event, EventType
from ..strategy.components.signal_replay import SignalReplayer, WeightedSignalAggregator
from ..risk.protocols import Signal
from .backtest_broker import BacktestBroker
from .execution_engine import DefaultExecutionEngine
from .market_simulation import MarketSimulator


class SignalReplayContainer(UniversalScopedContainer):
    """
    Container for signal replay and ensemble optimization.
    
    This container:
    - Reads captured signals from Phase 1
    - Applies ensemble weights to combine signals
    - NO indicator computation (already done in Phase 1)
    - NO classifier needed (regime context in signals)
    - Processes through Risk & Portfolio for position sizing
    - Executes orders through standard backtest engine
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        signal_log_path: Optional[str] = None,
        shared_services: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize signal replay container.
        
        Args:
            container_id: Unique container ID
            signal_log_path: Path to signal log file
            shared_services: Shared read-only services
        """
        # Generate structured container ID
        if container_id is None:
            container_id = f"signal_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
        super().__init__(
            container_id=container_id,
            container_type="signal_replay",
            shared_services=shared_services
        )
        
        self.signal_log_path = signal_log_path
        
        # Core components
        self.signal_replayer: Optional[SignalReplayer] = None
        self.ensemble_optimizer: Optional[WeightedSignalAggregator] = None
        self.risk_portfolio_container = None
        self.execution_engine: Optional[DefaultExecutionEngine] = None
        self.backtest_broker: Optional[BacktestBroker] = None
        
        # Performance tracking
        self._processed_signals = 0
        self._generated_orders = 0
        self._executed_fills = 0
        
    def setup_components(self, config: Dict[str, Any]) -> None:
        """Set up container components based on configuration."""
        # Create signal replayer
        self._create_signal_replayer(config.get('signal_config', {}))
        
        # Create ensemble optimizer
        self._create_ensemble_optimizer(config.get('weight_config', {}))
        
        # Create risk & portfolio container
        self._create_risk_portfolio_container(config.get('risk_config', {}))
        
        # Create execution components
        self._create_execution_components(config.get('execution_config', {}))
        
        # Wire up event flows
        self._wire_signal_replay_flows()
        
    def _create_signal_replayer(self, signal_config: Dict[str, Any]) -> None:
        """Create signal replayer component."""
        # Load signals from file if path provided
        signals = []
        if self.signal_log_path and Path(self.signal_log_path).exists():
            with open(self.signal_log_path, 'r') as f:
                signal_data = json.load(f)
                signals = signal_data if isinstance(signal_data, list) else []
                
        self.signal_replayer = SignalReplayer(
            captured_signals=signals,
            filter_criteria=signal_config.get('filter_criteria', {})
        )
        
        self.logger.info(f"Loaded {len(signals)} signals for replay")
        
    def _create_ensemble_optimizer(self, weight_config: Dict[str, Any]) -> None:
        """Create ensemble weight optimizer."""
        self.ensemble_optimizer = WeightedSignalAggregator(
            strategy_weights=weight_config.get('strategy_weights', {}),
            aggregation_method=weight_config.get('aggregation_method', 'weighted_vote'),
            min_agreement=weight_config.get('min_agreement', 0.5)
        )
        
    def _create_risk_portfolio_container(self, risk_config: Dict[str, Any]) -> None:
        """Create risk & portfolio container."""
        # Create sub-container for risk & portfolio
        risk_container_id = f"{self.container_id}_risk_portfolio"
        self.risk_portfolio_container = self.create_subcontainer(
            container_id=risk_container_id,
            container_type="risk_portfolio"
        )
        
        # Create risk manager component
        risk_spec = ComponentSpec(
            name="risk_manager",
            class_name=risk_config.get('risk_class', 'DefaultRiskManager'),
            parameters=risk_config.get('risk_parameters', {
                'max_position_size': 0.02,
                'max_total_exposure': 0.1
            }),
            capabilities=['event_publisher', 'event_subscriber']
        )
        
        self.risk_portfolio_container.create_component(risk_spec)
        
        # Create portfolio component
        portfolio_spec = ComponentSpec(
            name="portfolio",
            class_name=risk_config.get('portfolio_class', 'Portfolio'),
            parameters=risk_config.get('portfolio_parameters', {
                'initial_capital': 100000
            }),
            capabilities=['event_publisher']
        )
        
        self.risk_portfolio_container.create_component(portfolio_spec)
        
    def _create_execution_components(self, execution_config: Dict[str, Any]) -> None:
        """Create execution engine and broker."""
        # Create backtest broker
        self.backtest_broker = BacktestBroker(
            initial_balance=execution_config.get('initial_balance', 100000)
        )
        
        # Create market simulator
        market_sim = MarketSimulator(
            slippage_model=execution_config.get('slippage_model'),
            commission_model=execution_config.get('commission_model')
        )
        
        # Create execution engine
        self.execution_engine = DefaultExecutionEngine(
            broker=self.backtest_broker,
            market_simulator=market_sim
        )
        
    def _wire_signal_replay_flows(self) -> None:
        """Wire up the signal replay event flows."""
        # Signal flow: Replayer -> Ensemble -> Risk -> Execution
        
        # Subscribe risk container to aggregated signals
        if self.risk_portfolio_container:
            self.risk_portfolio_container.event_bus.subscribe(
                EventType.SIGNAL,
                self._handle_aggregated_signal
            )
            
        # Subscribe execution to orders
        self.event_bus.subscribe(
            EventType.ORDER,
            self.execution_engine.process_event
        )
        
        # Subscribe to fills for portfolio updates
        self.event_bus.subscribe(
            EventType.FILL,
            self._handle_fill_event
        )
        
    def _handle_aggregated_signal(self, event: Event) -> None:
        """Handle aggregated signals from ensemble."""
        self._processed_signals += 1
        
        # Forward to risk & portfolio for order generation
        if self.risk_portfolio_container:
            # Risk & portfolio will convert to orders
            self.risk_portfolio_container.event_bus.publish(event)
            
    def _handle_fill_event(self, event: Event) -> None:
        """Handle fill events."""
        self._executed_fills += 1
        
        # Update portfolio state
        if self.risk_portfolio_container:
            self.risk_portfolio_container.event_bus.publish(event)
            
    async def run_replay(
        self,
        start_date: datetime,
        end_date: datetime,
        market_data_source: Any = None
    ) -> Dict[str, Any]:
        """
        Run signal replay with ensemble weights.
        
        Args:
            start_date: Start date for replay
            end_date: End date for replay
            market_data_source: Optional market data for execution
            
        Returns:
            Backtest results
        """
        start_time = datetime.now()
        
        # Initialize components
        await self.initialize()
        await self.start()
        
        self.logger.info(
            f"Starting signal replay",
            start_date=start_date,
            end_date=end_date,
            ensemble_weights=self.ensemble_optimizer.strategy_weights
        )
        
        # Get signals to replay
        signals_to_replay = self.signal_replayer.get_signals_in_range(
            start_date, end_date
        )
        
        # Group signals by timestamp
        signals_by_time = {}
        for signal in signals_to_replay:
            timestamp = signal.get('timestamp')
            if timestamp not in signals_by_time:
                signals_by_time[timestamp] = []
            signals_by_time[timestamp].append(signal)
            
        # Process signals chronologically
        for timestamp in sorted(signals_by_time.keys()):
            timestamp_signals = signals_by_time[timestamp]
            
            # Apply ensemble weights
            aggregated_signals = self.ensemble_optimizer.aggregate_signals(
                timestamp_signals
            )
            
            # Publish aggregated signals
            for signal in aggregated_signals:
                self.event_bus.publish(Event(
                    event_type=EventType.SIGNAL,
                    payload={'signal': signal},
                    source_id="ensemble_optimizer"
                ))
                
            # Get market data if available
            if market_data_source:
                market_data = await market_data_source.get_data(timestamp)
                
                # Update execution context with market data
                self.execution_engine.update_market_data(market_data)
                
        # Calculate results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get performance metrics
        portfolio_value = self.backtest_broker.get_account_value()
        trades = self.backtest_broker.get_trade_history()
        
        results = {
            'container_id': self.container_id,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'duration_seconds': duration,
            'signals_processed': self._processed_signals,
            'orders_generated': self._generated_orders,
            'fills_executed': self._executed_fills,
            'final_portfolio_value': portfolio_value,
            'total_trades': len(trades),
            'ensemble_weights': self.ensemble_optimizer.strategy_weights,
            'performance_metrics': self._calculate_performance_metrics(trades)
        }
        
        # Clean up
        await self.stop()
        
        return results
        
    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
            
        # Calculate metrics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        return {
            'total_return': sum(t.get('pnl', 0) for t in trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'max_drawdown': self._calculate_max_drawdown(trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades)
        }
        
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0
            
        equity_curve = []
        running_pnl = 0
        
        for trade in sorted(trades, key=lambda x: x.get('exit_time', x.get('entry_time'))):
            running_pnl += trade.get('pnl', 0)
            equity_curve.append(running_pnl)
            
        if not equity_curve:
            return 0
            
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Sharpe ratio from trades."""
        if len(trades) < 2:
            return 0
            
        returns = [t.get('pnl', 0) for t in trades]
        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        
        return mean_return / std_return if std_return > 0 else 0


class SignalReplayContainerFactory:
    """Factory for creating signal replay containers."""
    
    @staticmethod
    def create_instance(config: Dict[str, Any]) -> SignalReplayContainer:
        """
        Create a signal replay container instance.
        
        Args:
            config: Container configuration including:
                - signal_log_path: Path to captured signals
                - weight_config: Ensemble weight configuration
                - risk_config: Risk & portfolio configuration
                - execution_config: Execution configuration
                
        Returns:
            Configured SignalReplayContainer
        """
        # Extract configuration
        container_id = config.get('container_id')
        signal_log_path = config.get('signal_log_path')
        
        # Create container
        container = SignalReplayContainer(
            container_id=container_id,
            signal_log_path=signal_log_path,
            shared_services=config.get('shared_services', {})
        )
        
        # Set up components
        container.setup_components(config)
        
        return container
        
    @staticmethod
    def create_for_ensemble_optimization(
        signal_log_path: str,
        strategy_weights: Dict[str, float],
        risk_parameters: Dict[str, Any],
        execution_config: Optional[Dict[str, Any]] = None
    ) -> SignalReplayContainer:
        """
        Create container specifically for ensemble weight optimization.
        
        Args:
            signal_log_path: Path to captured signals from Phase 1
            strategy_weights: Weights for each strategy
            risk_parameters: Risk management parameters
            execution_config: Execution configuration
            
        Returns:
            Configured SignalReplayContainer
        """
        config = {
            'signal_log_path': signal_log_path,
            'weight_config': {
                'strategy_weights': strategy_weights,
                'aggregation_method': 'weighted_vote',
                'min_agreement': 0.5
            },
            'risk_config': {
                'risk_parameters': risk_parameters
            },
            'execution_config': execution_config or {
                'initial_balance': 100000,
                'slippage_model': None,
                'commission_model': None
            }
        }
        
        return SignalReplayContainerFactory.create_instance(config)
        
    @staticmethod
    def create_batch_for_optimization(
        signal_log_path: str,
        weight_combinations: List[Dict[str, float]],
        base_config: Dict[str, Any]
    ) -> List[SignalReplayContainer]:
        """
        Create multiple containers for parallel weight optimization.
        
        Args:
            signal_log_path: Path to captured signals
            weight_combinations: List of weight combinations to test
            base_config: Base configuration for all containers
            
        Returns:
            List of configured containers
        """
        containers = []
        
        for i, weights in enumerate(weight_combinations):
            config = base_config.copy()
            config['container_id'] = f"ensemble_opt_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config['signal_log_path'] = signal_log_path
            config['weight_config'] = {
                'strategy_weights': weights,
                'aggregation_method': base_config.get('aggregation_method', 'weighted_vote')
            }
            
            container = SignalReplayContainerFactory.create_instance(config)
            containers.append(container)
            
        return containers