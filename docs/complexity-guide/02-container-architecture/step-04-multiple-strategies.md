# Step 4: Multiple Strategies

**Status**: Container Architecture Step
**Complexity**: Medium-High
**Prerequisites**: [Step 3: Classifier Container](step-03-classifier-container.md) completed
**Architecture Ref**: [CONTAINER-HIERARCHY.md](../../architecture/02-CONTAINER-HIERARCHY.md#strategy-coordination)

## 🎯 Objective

Implement multiple strategy coordination:
- Run multiple strategies in parallel
- Each strategy in its own isolated container
- Coordinate signal aggregation
- Manage resource allocation
- Enable strategy performance comparison

## 📋 Required Reading

Before starting:
1. [Protocol + Composition](../../architecture/03-PROTOCOL-COMPOSITION.md)
2. [Strategy Module](../../strategy/README.md)
3. [Event Bus Isolation](../validation-framework/event-bus-isolation.md)

## 🏗️ Implementation Tasks

### 1. Strategy Coordinator Container

```python
# src/strategy/strategy_coordinator.py
class StrategyCoordinator:
    """
    Coordinates multiple strategy containers.
    Manages signal aggregation and resource allocation.
    """
    
    def __init__(self, container_id: str, config: StrategyCoordinatorConfig):
        self.container_id = container_id
        self.config = config
        
        # Create master event bus
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            f"{container_id}_coordinator"
        )
        
        # Strategy containers
        self.strategy_containers: Dict[str, StrategyContainer] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.performance_tracker = PerformanceTracker()
        
        # Signal aggregation
        self.signal_aggregator = SignalAggregator(config.aggregation_method)
        self.aggregated_signals: List[AggregatedSignal] = []
        
        # Setup logging
        self.logger = ComponentLogger("StrategyCoordinator", container_id)
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Create and configure strategy containers"""
        for strategy_id, strategy_config in self.config.strategies.items():
            # Create strategy container
            container = StrategyContainer(
                container_id=f"{self.container_id}_{strategy_id}",
                strategy_type=strategy_config['type'],
                params=strategy_config['params']
            )
            
            # Set initial weight
            self.strategy_weights[strategy_id] = strategy_config.get('weight', 1.0)
            
            # Subscribe to strategy signals
            container.event_bus.subscribe(
                "SIGNAL",
                lambda signal, sid=strategy_id: self.on_strategy_signal(sid, signal)
            )
            
            # Store container
            self.strategy_containers[strategy_id] = container
            
            self.logger.info(f"Initialized strategy: {strategy_id}")
    
    def on_bar(self, bar: Bar) -> None:
        """Distribute market data to all strategies"""
        # Clear previous signals
        self.aggregated_signals.clear()
        
        # Send to all strategies
        for strategy_id, container in self.strategy_containers.items():
            try:
                container.on_bar(bar)
            except Exception as e:
                self.logger.error(
                    f"Strategy {strategy_id} failed on bar: {e}"
                )
                # Continue with other strategies
        
        # Aggregate signals if any received
        if self.aggregated_signals:
            self._process_aggregated_signals(bar.timestamp)
    
    def on_strategy_signal(self, strategy_id: str, signal: TradingSignal) -> None:
        """Handle signal from individual strategy"""
        self.logger.log_event_flow(
            "SIGNAL_RECEIVED", f"strategy_{strategy_id}", "coordinator",
            f"Signal: {signal.direction} {signal.symbol}"
        )
        
        # Track signal
        self.aggregated_signals.append(
            AggregatedSignal(
                strategy_id=strategy_id,
                signal=signal,
                weight=self.strategy_weights[strategy_id],
                timestamp=datetime.now()
            )
        )
    
    def _process_aggregated_signals(self, timestamp: datetime) -> None:
        """Aggregate and emit consensus signal"""
        # Group by symbol
        signals_by_symbol = self._group_signals_by_symbol()
        
        for symbol, signals in signals_by_symbol.items():
            # Aggregate signals for this symbol
            consensus = self.signal_aggregator.aggregate(signals)
            
            if consensus:
                # Log aggregation details
                self.logger.info(
                    f"Consensus signal for {symbol}: {consensus.direction} "
                    f"(confidence: {consensus.confidence:.2f})"
                )
                
                # Emit aggregated signal
                self.event_bus.publish("AGGREGATED_SIGNAL", consensus)
```

### 2. Signal Aggregation Methods

```python
# src/strategy/signal_aggregation.py
from abc import ABC, abstractmethod

class SignalAggregator(ABC):
    """Base class for signal aggregation methods"""
    
    @abstractmethod
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        pass

class WeightedVotingAggregator(SignalAggregator):
    """Weighted voting signal aggregation"""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
    
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        """Aggregate signals using weighted voting"""
        if not signals:
            return None
        
        # Calculate weighted votes
        buy_weight = sum(
            s.weight for s in signals 
            if s.signal.direction == Direction.BUY
        )
        sell_weight = sum(
            s.weight for s in signals 
            if s.signal.direction == Direction.SELL
        )
        total_weight = sum(s.weight for s in signals)
        
        # Determine consensus
        if buy_weight > sell_weight:
            direction = Direction.BUY
            confidence = buy_weight / total_weight
        elif sell_weight > buy_weight:
            direction = Direction.SELL
            confidence = sell_weight / total_weight
        else:
            return None  # No consensus
        
        # Check minimum confidence
        if confidence < self.min_confidence:
            return None
        
        # Calculate strength as weighted average
        relevant_signals = [
            s for s in signals if s.signal.direction == direction
        ]
        weighted_strength = sum(
            s.signal.strength * s.weight for s in relevant_signals
        ) / sum(s.weight for s in relevant_signals)
        
        return ConsensusSignal(
            symbol=signals[0].signal.symbol,
            direction=direction,
            strength=weighted_strength,
            confidence=confidence,
            contributing_strategies=[s.strategy_id for s in relevant_signals],
            timestamp=datetime.now()
        )

class MajorityVotingAggregator(SignalAggregator):
    """Simple majority voting aggregation"""
    
    def __init__(self, min_agreement: float = 0.6):
        self.min_agreement = min_agreement
    
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        """Require majority agreement"""
        if not signals:
            return None
        
        # Count votes
        buy_count = sum(1 for s in signals if s.signal.direction == Direction.BUY)
        sell_count = sum(1 for s in signals if s.signal.direction == Direction.SELL)
        total_count = len(signals)
        
        # Check for majority
        if buy_count / total_count >= self.min_agreement:
            return self._create_consensus(signals, Direction.BUY, buy_count / total_count)
        elif sell_count / total_count >= self.min_agreement:
            return self._create_consensus(signals, Direction.SELL, sell_count / total_count)
        
        return None
```

### 3. Performance Tracking

```python
# src/strategy/performance_tracking.py
class StrategyPerformanceTracker:
    """Tracks individual strategy performance for weight adjustment"""
    
    def __init__(self):
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.rolling_window = 100  # trades
    
    def update_on_fill(self, strategy_id: str, fill: Fill) -> None:
        """Update strategy metrics based on fill"""
        if strategy_id not in self.strategy_metrics:
            self.strategy_metrics[strategy_id] = StrategyMetrics(strategy_id)
        
        metrics = self.strategy_metrics[strategy_id]
        metrics.add_fill(fill)
        
        # Update rolling statistics
        if metrics.trade_count >= self.rolling_window:
            self._update_rolling_stats(metrics)
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on performance"""
        if not self.strategy_metrics:
            return {}
        
        # Calculate Sharpe ratios
        sharpe_ratios = {
            sid: metrics.calculate_sharpe_ratio() 
            for sid, metrics in self.strategy_metrics.items()
        }
        
        # Convert to weights (only positive Sharpe)
        positive_sharpes = {
            sid: max(0, sharpe) 
            for sid, sharpe in sharpe_ratios.items()
        }
        
        total_sharpe = sum(positive_sharpes.values())
        
        if total_sharpe == 0:
            # Equal weights if all negative
            return {sid: 1.0 / len(self.strategy_metrics) 
                   for sid in self.strategy_metrics}
        
        # Weight by Sharpe ratio
        return {
            sid: sharpe / total_sharpe 
            for sid, sharpe in positive_sharpes.items()
        }
```

### 4. Strategy Container Implementation

```python
# src/strategy/strategy_container.py
class StrategyContainer:
    """Container for individual strategy with lifecycle management"""
    
    def __init__(self, container_id: str, strategy_type: str, params: Dict):
        self.container_id = container_id
        self.strategy_type = strategy_type
        self.params = params
        
        # Create isolated components
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            f"{container_id}_strategy"
        )
        
        # Create strategy
        self.strategy = self._create_strategy(strategy_type, params)
        self.is_active = True
        
        # Performance metrics
        self.signals_generated = 0
        self.last_signal_time = None
        
        # Setup logging
        self.logger = ComponentLogger("StrategyContainer", container_id)
    
    def _create_strategy(self, strategy_type: str, params: Dict) -> BaseStrategy:
        """Factory method for strategy creation"""
        strategy_map = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'trend_following': TrendFollowingStrategy,
            'pairs_trading': PairsTradingStrategy
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = strategy_map[strategy_type]
        return strategy_class(**params)
    
    def on_bar(self, bar: Bar) -> None:
        """Process market data"""
        if not self.is_active:
            return
        
        try:
            # Generate signal
            signal = self.strategy.generate_signal(bar)
            
            if signal:
                self.signals_generated += 1
                self.last_signal_time = bar.timestamp
                
                # Emit signal
                self.event_bus.publish("SIGNAL", signal)
                
                self.logger.info(
                    f"Signal generated: {signal.direction} {signal.symbol} "
                    f"strength={signal.strength:.2f}"
                )
        except Exception as e:
            self.logger.error(f"Strategy error: {e}")
            raise
    
    def deactivate(self) -> None:
        """Deactivate strategy"""
        self.is_active = False
        self.strategy.cleanup()
        self.logger.info("Strategy deactivated")
    
    def get_state(self) -> Dict:
        """Get strategy state for persistence"""
        return {
            'container_id': self.container_id,
            'strategy_type': self.strategy_type,
            'params': self.params,
            'signals_generated': self.signals_generated,
            'is_active': self.is_active,
            'strategy_state': self.strategy.get_state()
        }
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step4_multiple_strategies.py`:

```python
class TestSignalAggregation:
    """Test signal aggregation methods"""
    
    def test_weighted_voting(self):
        """Test weighted voting aggregation"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        # Create test signals
        signals = [
            AggregatedSignal(
                strategy_id="momentum",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8),
                weight=0.4
            ),
            AggregatedSignal(
                strategy_id="trend",
                signal=TradingSignal("AAPL", Direction.BUY, 0.9),
                weight=0.3
            ),
            AggregatedSignal(
                strategy_id="mean_rev",
                signal=TradingSignal("AAPL", Direction.SELL, 0.7),
                weight=0.3
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.confidence == 0.7  # (0.4 + 0.3) / 1.0
        assert len(consensus.contributing_strategies) == 2
```

### Integration Tests

Create `tests/integration/test_step4_strategy_coordination.py`:

```python
def test_multiple_strategy_isolation():
    """Test strategies run in isolation"""
    config = {
        'strategies': {
            'momentum': {'type': 'momentum', 'params': {'period': 20}},
            'mean_rev': {'type': 'mean_reversion', 'params': {'period': 10}}
        },
        'aggregation_method': 'weighted_voting'
    }
    
    coordinator = StrategyCoordinator("test_coord", config)
    
    # Verify separate event buses
    momentum_bus = coordinator.strategy_containers['momentum'].event_bus
    mean_rev_bus = coordinator.strategy_containers['mean_rev'].event_bus
    
    assert momentum_bus != mean_rev_bus
    
    # Test no signal leakage
    signals_received = []
    momentum_bus.subscribe("SIGNAL", lambda s: signals_received.append(s))
    
    # Send bar to coordinator
    bar = create_test_bar()
    coordinator.on_bar(bar)
    
    # Only momentum signals should be received
    momentum_signals = [
        s for s in signals_received 
        if s.metadata.get('strategy_id') == 'momentum'
    ]
    assert len(momentum_signals) == len(signals_received)

def test_signal_aggregation_flow():
    """Test complete signal aggregation workflow"""
    coordinator = create_test_coordinator()
    
    # Capture aggregated signals
    aggregated = []
    coordinator.event_bus.subscribe(
        "AGGREGATED_SIGNAL", 
        lambda s: aggregated.append(s)
    )
    
    # Create data that triggers signals from multiple strategies
    bars = create_multi_signal_scenario()
    
    for bar in bars:
        coordinator.on_bar(bar)
    
    # Verify aggregation occurred
    assert len(aggregated) > 0
    assert all(s.confidence >= 0.6 for s in aggregated)
```

### System Tests

Create `tests/system/test_step4_multi_strategy_backtest.py`:

```python
def test_multi_strategy_system():
    """Test complete system with multiple strategies"""
    # Configure system
    config = {
        'strategies': {
            'momentum': {
                'type': 'momentum',
                'params': {'fast': 10, 'slow': 30},
                'weight': 0.4
            },
            'mean_reversion': {
                'type': 'mean_reversion',
                'params': {'period': 20, 'threshold': 2.0},
                'weight': 0.3
            },
            'trend_following': {
                'type': 'trend_following',
                'params': {'period': 50},
                'weight': 0.3
            }
        },
        'risk': {
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02
        }
    }
    
    # Run backtest
    data = create_diverse_market_data()
    system = create_multi_strategy_system(config)
    results = system.run_backtest(data)
    
    # Verify all strategies contributed
    strategy_contributions = results['strategy_contributions']
    assert all(count > 0 for count in strategy_contributions.values())
    
    # Verify performance
    assert results['sharpe_ratio'] > 1.2
    assert results['max_drawdown'] < 0.10
    
    # Verify signal aggregation worked
    assert results['aggregated_signals_count'] > 0
    assert results['consensus_rate'] > 0.3  # At least 30% consensus
```

## ✅ Validation Checklist

### Strategy Coordination
- [ ] Multiple strategies run concurrently
- [ ] Each strategy properly isolated
- [ ] Signal aggregation working
- [ ] Performance tracking accurate

### Resource Management
- [ ] Memory usage scales linearly
- [ ] CPU usage distributed fairly
- [ ] No strategy starvation
- [ ] Graceful error handling

### Testing Validation
- [ ] Unit tests for aggregation methods
- [ ] Integration tests for coordination
- [ ] System tests with multiple strategies
- [ ] Performance within targets

## 📊 Memory & Performance

### Memory Monitoring
```python
@profile
def run_multi_strategy_backtest(coordinator, data):
    """Profile memory usage with multiple strategies"""
    initial_memory = get_memory_usage()
    
    for i, bar in enumerate(data):
        coordinator.on_bar(bar)
        
        if i % 1000 == 0:
            current_memory = get_memory_usage()
            memory_growth = current_memory - initial_memory
            assert memory_growth < 100 * 1024 * 1024  # 100MB limit
```

### Performance Targets
- Process 5 strategies concurrently
- < 5ms total processing per bar
- < 100MB memory per strategy
- Linear scaling with strategy count

## 🐛 Common Issues

1. **Signal Timing Conflicts**
   - Ensure all strategies process same bar
   - Handle late signals appropriately
   - Consider signal expiration

2. **Weight Balancing**
   - Monitor strategy contribution rates
   - Adjust weights dynamically
   - Prevent single strategy dominance

3. **Error Propagation**
   - Isolate strategy failures
   - Continue with working strategies
   - Log and monitor failures

## 🎯 Success Criteria

Step 4 is complete when:
1. ✅ Multiple strategies run in isolation
2. ✅ Signal aggregation produces consensus
3. ✅ Performance tracking implemented
4. ✅ Resource usage acceptable
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 5: Multiple Risk Containers](step-05-multiple-risk.md)

## 📚 Additional Resources

- [Strategy Design Patterns](../../strategy/patterns.md)
- [Signal Aggregation Theory](../references/signal-aggregation.md)
- [Performance Attribution](../references/performance-attribution.md)