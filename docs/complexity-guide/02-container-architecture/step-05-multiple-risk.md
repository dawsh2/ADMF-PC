# Step 5: Multiple Risk Containers

**Status**: Container Architecture Step
**Complexity**: High
**Prerequisites**: [Step 4: Multiple Strategies](step-04-multiple-strategies.md) completed
**Architecture Ref**: [CONTAINER-HIERARCHY.md](../../architecture/02-CONTAINER-HIERARCHY.md#risk-isolation)

## 🎯 Objective

Implement multiple risk containers for:
- Strategy-specific risk management
- Isolated risk limits per strategy
- Portfolio-level risk aggregation
- Risk budget allocation
- Cross-strategy correlation monitoring

## 📋 Required Reading

Before starting:
1. [Risk Module Architecture](../../risk/README.md)
2. [Container Isolation](../validation-framework/event-bus-isolation.md)
3. [Portfolio State Management](../../risk/portfolio_state.py)

## 🏗️ Implementation Tasks

### 0. Container Hierarchy Inversion Experiment

Before implementing the standard hierarchy, experiment with **inverted container organization** to optimize for risk manager comparison:

**Standard Hierarchy** (many containers):
```
Classifier/Risk/Portfolio/Strategy
├── HMM Classifier
│   ├── Conservative Risk (Strategy A + Strategy B + Strategy C)
│   ├── Balanced Risk (Strategy A + Strategy B + Strategy C)  
│   └── Aggressive Risk (Strategy A + Strategy B + Strategy C)
└── Pattern Classifier
    ├── Conservative Risk (Strategy A + Strategy B + Strategy C)
    ├── Balanced Risk (Strategy A + Strategy B + Strategy C)
    └── Aggressive Risk (Strategy A + Strategy B + Strategy C)

Total Containers: 2 classifiers × 3 risk × 3 strategies = 18 containers
```

**Inverted Hierarchy** (fewer containers):
```
Classifier/Strategy/Portfolio/Risk
├── HMM Classifier
│   ├── Strategy A (Conservative + Balanced + Aggressive Risk)
│   ├── Strategy B (Conservative + Balanced + Aggressive Risk)
│   └── Strategy C (Conservative + Balanced + Aggressive Risk)
└── Pattern Classifier  
    ├── Strategy A (Conservative + Balanced + Aggressive Risk)
    ├── Strategy B (Conservative + Balanced + Aggressive Risk)
    └── Strategy C (Conservative + Balanced + Aggressive Risk)

Total Containers: 2 classifiers × 3 strategies = 6 containers
Each container tests 3 risk managers
```

#### Why Hierarchy Inversion Makes Sense:

1. **Performance Attribution**: "How does Strategy A perform with different risk managers?"
2. **Risk Manager A/B Testing**: Keep strategy logic constant, vary risk parameters
3. **Resource Efficiency**: 6 containers instead of 18 for same comparison
4. **Cleaner Analysis**: Direct comparison of risk manager effectiveness per strategy

#### Adapter Configuration for Inverted Hierarchy:

```yaml
# Inverted hierarchy adapter configuration
communication:
  adapters:
    # Classifier to Strategy containers (unchanged)
    - type: hierarchical
      parent: hmm_classifier
      children: [strategy_a_container, strategy_b_container, strategy_c_container]
      
    # Strategy to multiple risk managers (NEW PATTERN)
    - type: broadcast
      source: strategy_a_signals
      targets: [conservative_risk_a, balanced_risk_a, aggressive_risk_a]
      transform: add_strategy_context
      
    # Risk managers aggregate back to strategy container
    - type: aggregation  # NEW ADAPTER TYPE
      sources: [conservative_risk_a, balanced_risk_a, aggressive_risk_a]
      target: strategy_a_results
      aggregation_method: risk_comparison
```

#### Implementation:

```python
# src/risk/inverted_hierarchy_manager.py
class InvertedHierarchyRiskManager:
    """
    Manages multiple risk managers within a single strategy container.
    Enables direct comparison of risk management approaches.
    """
    
    def __init__(self, strategy_id: str, risk_configs: Dict[str, RiskConfig]):
        self.strategy_id = strategy_id
        self.risk_managers: Dict[str, RiskManager] = {}
        self.risk_results: Dict[str, RiskResults] = {}
        
        # Create multiple risk managers for this strategy
        for risk_id, config in risk_configs.items():
            self.risk_managers[risk_id] = RiskManager(
                container_id=f"{strategy_id}_{risk_id}",
                config=config
            )
            
        self.logger = ComponentLogger("InvertedRiskManager", strategy_id)
    
    def process_strategy_signal(self, signal: TradingSignal) -> Dict[str, Order]:
        """Process same signal through multiple risk managers"""
        orders = {}
        
        for risk_id, risk_manager in self.risk_managers.items():
            try:
                # Clone signal for isolation
                signal_copy = signal.copy()
                order = risk_manager.process_signal(signal_copy)
                
                if order:
                    orders[risk_id] = order
                    self.logger.info(
                        f"Risk manager {risk_id} generated order: "
                        f"{order.side} {order.quantity} {order.symbol}"
                    )
                else:
                    self.logger.info(f"Risk manager {risk_id} rejected signal")
                    
            except Exception as e:
                self.logger.error(f"Risk manager {risk_id} failed: {e}")
        
        return orders
    
    def get_comparative_results(self) -> Dict[str, Any]:
        """Get side-by-side risk manager performance"""
        results = {}
        
        for risk_id, risk_manager in self.risk_managers.items():
            metrics = risk_manager.get_performance_metrics()
            results[risk_id] = {
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'total_return': metrics.total_return,
                'trade_count': metrics.trade_count,
                'win_rate': metrics.win_rate
            }
        
        return {
            'strategy_id': self.strategy_id,
            'risk_manager_comparison': results,
            'best_risk_manager': self._identify_best_risk_manager(results)
        }
```

#### Benefits of Hierarchy Inversion:

**Resource Efficiency:**
```
Standard: 2 × 3 × 3 = 18 containers (heavy)
Inverted: 2 × 3 = 6 containers (light)
```

**Cleaner Comparisons:**
```python
# Direct risk manager comparison per strategy
strategy_a_results = {
    'conservative': {'sharpe': 1.2, 'drawdown': 0.05},
    'balanced': {'sharpe': 1.5, 'drawdown': 0.08}, 
    'aggressive': {'sharpe': 1.8, 'drawdown': 0.15}
}
# Winner: Aggressive risk for Strategy A
```

**Easier Configuration:**
```yaml
# Test same strategy with different risk approaches
strategy_a:
  type: momentum
  risk_managers:
    conservative: {max_position: 0.02, stop_loss: 0.01}
    balanced: {max_position: 0.05, stop_loss: 0.02}
    aggressive: {max_position: 0.10, stop_loss: 0.05}
```

#### When to Use Each Hierarchy:

**Use Inverted (Classifier/Strategy/Risk) When:**
- Comparing risk management approaches
- Strategy logic is fixed/proven
- Optimizing risk parameters
- A/B testing risk managers

**Use Standard (Classifier/Risk/Strategy) When:**
- Comparing strategy performance
- Risk approach is fixed/proven  
- Optimizing strategy parameters
- A/B testing strategies

#### Implementation Note:

This substep validates that your adapter system can handle **arbitrary container hierarchies** - the same components, just reorganized. This proves the architecture's flexibility and sets up powerful optimization patterns.

### 1. Risk Container Hierarchy

```python
# src/risk/hierarchical_risk_manager.py
class HierarchicalRiskManager:
    """
    Manages multiple risk containers with portfolio-level oversight.
    Each strategy gets its own risk container with allocated capital.
    """
    
    def __init__(self, container_id: str, config: HierarchicalRiskConfig):
        self.container_id = container_id
        self.config = config
        
        # Portfolio-level state
        self.total_capital = config.total_capital
        self.risk_containers: Dict[str, RiskContainer] = {}
        self.capital_allocations: Dict[str, float] = {}
        
        # Cross-strategy monitoring
        self.correlation_monitor = CorrelationMonitor()
        self.portfolio_risk_limits = PortfolioRiskLimits(config.portfolio_limits)
        
        # Master event bus for portfolio-level events
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            f"{container_id}_portfolio_risk"
        )
        
        # Setup logging
        self.logger = ComponentLogger("HierarchicalRiskManager", container_id)
        
        # Initialize risk containers
        self._initialize_risk_containers()
    
    def _initialize_risk_containers(self) -> None:
        """Create risk container for each strategy"""
        for strategy_id, allocation in self.config.allocations.items():
            # Calculate allocated capital
            allocated_capital = self.total_capital * allocation['weight']
            
            # Create risk config for this strategy
            risk_config = RiskConfig(
                initial_capital=allocated_capital,
                max_position_size=allocation.get('max_position_size', 0.1),
                max_drawdown=allocation.get('max_drawdown', 0.2),
                risk_per_trade=allocation.get('risk_per_trade', 0.02)
            )
            
            # Create isolated risk container
            risk_container = RiskContainer(
                container_id=f"{self.container_id}_{strategy_id}_risk",
                config=risk_config
            )
            
            # Subscribe to risk events
            risk_container.event_bus.subscribe(
                "RISK_LIMIT_BREACH",
                lambda event, sid=strategy_id: self.on_risk_limit_breach(sid, event)
            )
            
            # Store container and allocation
            self.risk_containers[strategy_id] = risk_container
            self.capital_allocations[strategy_id] = allocated_capital
            
            self.logger.info(
                f"Initialized risk container for {strategy_id} "
                f"with capital: ${allocated_capital:,.2f}"
            )
    
    def process_strategy_signal(self, strategy_id: str, signal: TradingSignal) -> None:
        """Process signal through strategy-specific risk container"""
        if strategy_id not in self.risk_containers:
            self.logger.error(f"Unknown strategy: {strategy_id}")
            return
        
        # Get strategy's risk container
        risk_container = self.risk_containers[strategy_id]
        
        # Check portfolio-level constraints first
        if not self._check_portfolio_constraints(strategy_id, signal):
            self.logger.warning(
                f"Signal from {strategy_id} rejected by portfolio constraints"
            )
            return
        
        # Process through strategy's risk container
        risk_container.on_signal(signal)
    
    def _check_portfolio_constraints(self, strategy_id: str, 
                                   signal: TradingSignal) -> bool:
        """Check portfolio-level risk constraints"""
        # Get current portfolio state
        portfolio_state = self._aggregate_portfolio_state()
        
        # Check correlation limits
        if not self.correlation_monitor.check_correlation_limit(
            strategy_id, signal, portfolio_state
        ):
            return False
        
        # Check total exposure limits
        if not self.portfolio_risk_limits.check_exposure_limit(
            portfolio_state, signal
        ):
            return False
        
        # Check concentration limits
        if not self.portfolio_risk_limits.check_concentration_limit(
            portfolio_state, signal
        ):
            return False
        
        return True
```

### 2. Cross-Strategy Correlation Monitoring

```python
# src/risk/correlation_monitor.py
class CorrelationMonitor:
    """
    Monitors correlations between strategy positions.
    Prevents excessive correlation risk.
    """
    
    def __init__(self, max_correlation: float = 0.7, window: int = 252):
        self.max_correlation = max_correlation
        self.window = window
        self.returns_history: Dict[str, deque] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.logger = ComponentLogger("CorrelationMonitor", "portfolio")
    
    def update_returns(self, strategy_returns: Dict[str, float]) -> None:
        """Update returns history for correlation calculation"""
        timestamp = datetime.now()
        
        for strategy_id, return_value in strategy_returns.items():
            if strategy_id not in self.returns_history:
                self.returns_history[strategy_id] = deque(maxlen=self.window)
            
            self.returns_history[strategy_id].append({
                'timestamp': timestamp,
                'return': return_value
            })
        
        # Recalculate correlation matrix if enough data
        if self._has_sufficient_data():
            self._update_correlation_matrix()
    
    def _update_correlation_matrix(self) -> None:
        """Calculate correlation matrix between strategies"""
        # Convert to DataFrame
        returns_data = {}
        for strategy_id, history in self.returns_history.items():
            if len(history) >= 20:  # Minimum for correlation
                returns_data[strategy_id] = [h['return'] for h in history]
        
        if len(returns_data) >= 2:
            df = pd.DataFrame(returns_data)
            self.correlation_matrix = df.corr()
            
            # Log high correlations
            self._log_high_correlations()
    
    def check_correlation_limit(self, strategy_id: str, signal: TradingSignal,
                              portfolio_state: PortfolioState) -> bool:
        """Check if new position would violate correlation limits"""
        if self.correlation_matrix.empty:
            return True  # No correlation data yet
        
        # Get strategies with existing positions in same symbol
        strategies_in_symbol = [
            s for s, pos in portfolio_state.positions_by_strategy.items()
            if signal.symbol in pos and s != strategy_id
        ]
        
        # Check correlations
        for other_strategy in strategies_in_symbol:
            if strategy_id in self.correlation_matrix and \
               other_strategy in self.correlation_matrix:
                correlation = self.correlation_matrix.loc[strategy_id, other_strategy]
                
                if abs(correlation) > self.max_correlation:
                    self.logger.warning(
                        f"High correlation ({correlation:.2f}) between "
                        f"{strategy_id} and {other_strategy}"
                    )
                    return False
        
        return True
```

### 3. Dynamic Capital Reallocation

```python
# src/risk/capital_allocator.py
class DynamicCapitalAllocator:
    """
    Dynamically reallocates capital based on strategy performance.
    Uses Kelly criterion with safety factor.
    """
    
    def __init__(self, rebalance_frequency: int = 20):
        self.rebalance_frequency = rebalance_frequency
        self.trade_count = 0
        self.performance_window = 100
        self.safety_factor = 0.25  # Use 25% of Kelly
        self.min_allocation = 0.05  # 5% minimum
        self.max_allocation = 0.40  # 40% maximum
    
    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance"""
        self.trade_count += 1
        return self.trade_count % self.rebalance_frequency == 0
    
    def calculate_optimal_allocations(self, 
                                    strategy_metrics: Dict[str, StrategyMetrics]
                                    ) -> Dict[str, float]:
        """Calculate optimal capital allocations using Kelly criterion"""
        kelly_fractions = {}
        
        for strategy_id, metrics in strategy_metrics.items():
            # Calculate Kelly fraction
            win_rate = metrics.win_rate
            avg_win = metrics.avg_win
            avg_loss = abs(metrics.avg_loss)
            
            if avg_loss > 0 and win_rate > 0:
                # Kelly formula: f = (p * b - q) / b
                # where p = win_rate, q = 1-p, b = avg_win/avg_loss
                b = avg_win / avg_loss
                kelly = (win_rate * b - (1 - win_rate)) / b
                
                # Apply safety factor
                safe_kelly = kelly * self.safety_factor
                
                # Apply bounds
                kelly_fractions[strategy_id] = np.clip(
                    safe_kelly, self.min_allocation, self.max_allocation
                )
            else:
                kelly_fractions[strategy_id] = self.min_allocation
        
        # Normalize to sum to 1.0
        total = sum(kelly_fractions.values())
        return {
            sid: fraction / total 
            for sid, fraction in kelly_fractions.items()
        }
    
    def reallocate_capital(self, risk_manager: HierarchicalRiskManager,
                         new_allocations: Dict[str, float]) -> None:
        """Reallocate capital between strategies"""
        total_capital = risk_manager.total_capital
        
        for strategy_id, new_weight in new_allocations.items():
            old_capital = risk_manager.capital_allocations[strategy_id]
            new_capital = total_capital * new_weight
            
            if abs(new_capital - old_capital) > 0.01:  # Significant change
                # Update allocation
                risk_manager.capital_allocations[strategy_id] = new_capital
                
                # Update risk container capital
                risk_container = risk_manager.risk_containers[strategy_id]
                risk_container.update_capital(new_capital)
                
                risk_manager.logger.info(
                    f"Reallocated {strategy_id}: "
                    f"${old_capital:,.2f} → ${new_capital:,.2f} "
                    f"({new_weight:.1%})"
                )
```

### 4. Portfolio Risk Aggregation

```python
# src/risk/portfolio_risk_aggregator.py
class PortfolioRiskAggregator:
    """
    Aggregates risk metrics across all strategies.
    Provides portfolio-level risk view.
    """
    
    def aggregate_portfolio_state(self, 
                                risk_containers: Dict[str, RiskContainer]
                                ) -> AggregatedPortfolioState:
        """Aggregate state from all risk containers"""
        # Collect positions
        all_positions = {}
        positions_by_strategy = {}
        
        for strategy_id, container in risk_containers.items():
            strategy_positions = container.portfolio_state.positions
            positions_by_strategy[strategy_id] = strategy_positions
            
            # Aggregate by symbol
            for symbol, position in strategy_positions.items():
                if symbol not in all_positions:
                    all_positions[symbol] = {
                        'total_quantity': 0,
                        'weighted_avg_price': 0,
                        'strategies': []
                    }
                
                all_positions[symbol]['total_quantity'] += position.quantity
                all_positions[symbol]['strategies'].append(strategy_id)
        
        # Calculate portfolio metrics
        total_value = sum(
            container.portfolio_state.total_value 
            for container in risk_containers.values()
        )
        
        total_cash = sum(
            container.portfolio_state.cash 
            for container in risk_containers.values()
        )
        
        return AggregatedPortfolioState(
            positions=all_positions,
            positions_by_strategy=positions_by_strategy,
            total_value=total_value,
            total_cash=total_cash,
            timestamp=datetime.now()
        )
    
    def calculate_portfolio_var(self, 
                              portfolio_state: AggregatedPortfolioState,
                              confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        # Implementation of parametric VaR
        # This is simplified - production would use more sophisticated methods
        pass
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step5_multiple_risk.py`:

```python
class TestHierarchicalRiskManager:
    """Test hierarchical risk management"""
    
    def test_capital_allocation(self):
        """Test initial capital allocation"""
        config = HierarchicalRiskConfig(
            total_capital=1000000,
            allocations={
                'momentum': {'weight': 0.4, 'max_position_size': 0.1},
                'mean_rev': {'weight': 0.3, 'max_position_size': 0.15},
                'trend': {'weight': 0.3, 'max_position_size': 0.1}
            }
        )
        
        risk_manager = HierarchicalRiskManager("test", config)
        
        assert risk_manager.capital_allocations['momentum'] == 400000
        assert risk_manager.capital_allocations['mean_rev'] == 300000
        assert risk_manager.capital_allocations['trend'] == 300000

class TestCorrelationMonitor:
    """Test correlation monitoring"""
    
    def test_correlation_calculation(self):
        """Test correlation matrix calculation"""
        monitor = CorrelationMonitor(max_correlation=0.7)
        
        # Add correlated returns
        for i in range(50):
            returns = {
                'strategy1': 0.01 * np.sin(i),
                'strategy2': 0.01 * np.sin(i) * 0.9,  # 90% correlated
                'strategy3': 0.01 * np.random.randn()  # Uncorrelated
            }
            monitor.update_returns(returns)
        
        # Check correlation matrix
        corr = monitor.correlation_matrix
        assert abs(corr.loc['strategy1', 'strategy2'] - 0.9) < 0.1
        assert abs(corr.loc['strategy1', 'strategy3']) < 0.3
```

### Integration Tests

Create `tests/integration/test_step5_risk_isolation.py`:

```python
def test_risk_container_isolation():
    """Test risk containers are properly isolated"""
    risk_manager = create_test_hierarchical_risk_manager()
    
    # Get two risk containers
    momentum_risk = risk_manager.risk_containers['momentum']
    mean_rev_risk = risk_manager.risk_containers['mean_rev']
    
    # Verify separate event buses
    assert momentum_risk.event_bus != mean_rev_risk.event_bus
    
    # Test position isolation
    signal1 = TradingSignal("AAPL", Direction.BUY, 0.8)
    risk_manager.process_strategy_signal('momentum', signal1)
    
    # Mean reversion shouldn't see momentum's position
    assert "AAPL" not in mean_rev_risk.portfolio_state.positions
    assert "AAPL" in momentum_risk.portfolio_state.positions

def test_portfolio_constraint_enforcement():
    """Test portfolio-level constraints"""
    risk_manager = create_test_hierarchical_risk_manager()
    
    # Create signals that would violate correlation limit
    signal1 = TradingSignal("AAPL", Direction.BUY, 0.9)
    signal2 = TradingSignal("AAPL", Direction.BUY, 0.9)
    
    # Process first signal
    risk_manager.process_strategy_signal('strategy1', signal1)
    
    # Add correlation data
    risk_manager.correlation_monitor.correlation_matrix = pd.DataFrame(
        [[1.0, 0.95], [0.95, 1.0]], 
        index=['strategy1', 'strategy2'],
        columns=['strategy1', 'strategy2']
    )
    
    # Second signal should be rejected
    orders_before = count_total_orders(risk_manager)
    risk_manager.process_strategy_signal('strategy2', signal2)
    orders_after = count_total_orders(risk_manager)
    
    assert orders_after == orders_before  # No new order
```

### System Tests

Create `tests/system/test_step5_multi_risk_system.py`:

```python
def test_complete_multi_risk_system():
    """Test complete system with multiple risk containers"""
    # Setup system
    config = {
        'total_capital': 1000000,
        'strategies': {
            'momentum': {
                'weight': 0.4,
                'max_drawdown': 0.15,
                'strategy_params': {...}
            },
            'mean_reversion': {
                'weight': 0.3,
                'max_drawdown': 0.10,
                'strategy_params': {...}
            },
            'pairs_trading': {
                'weight': 0.3,
                'max_drawdown': 0.20,
                'strategy_params': {...}
            }
        },
        'portfolio_limits': {
            'max_correlation': 0.7,
            'max_total_exposure': 0.8,
            'max_single_position': 0.15
        }
    }
    
    system = create_multi_risk_system(config)
    data = create_multi_regime_market_data()
    
    # Run backtest
    results = system.run_backtest(data)
    
    # Verify risk isolation
    strategy_drawdowns = results['strategy_drawdowns']
    assert strategy_drawdowns['momentum'] <= 0.15
    assert strategy_drawdowns['mean_reversion'] <= 0.10
    assert strategy_drawdowns['pairs_trading'] <= 0.20
    
    # Verify portfolio constraints
    assert results['max_correlation_reached'] <= 0.7
    assert results['max_exposure_reached'] <= 0.8
    
    # Verify capital reallocation occurred
    reallocation_events = results['capital_reallocations']
    assert len(reallocation_events) > 0
```

## ✅ Validation Checklist

### Risk Isolation
- [ ] Each strategy has isolated risk container
- [ ] No position state leakage
- [ ] Capital allocations enforced
- [ ] Independent risk limits

### Portfolio Coordination
- [ ] Correlation monitoring working
- [ ] Portfolio constraints enforced
- [ ] Risk aggregation accurate
- [ ] Dynamic reallocation functional

### Performance Requirements
- [ ] Risk checks < 2ms per signal
- [ ] Correlation calculation < 10ms
- [ ] Memory usage scales linearly
- [ ] No memory leaks

## 📊 Memory & Performance

### Memory Optimization
```python
class MemoryEfficientRiskContainer:
    """Optimized risk container for memory efficiency"""
    
    def __init__(self, config):
        # Use slots to reduce memory overhead
        __slots__ = ['positions', 'cash', 'pending_orders']
        
        # Limit history buffers
        self.max_history_size = 1000
        self.position_history = deque(maxlen=self.max_history_size)
```

### Performance Benchmarks
- 10 risk containers: < 5ms total per bar
- Correlation matrix update: < 10ms
- Capital reallocation: < 50ms
- Memory per container: < 20MB

## 🐛 Common Issues

1. **Capital Drift**
   - Track capital precisely
   - Handle partial fills correctly
   - Reconcile regularly

2. **Correlation Window**
   - Balance responsiveness vs stability
   - Handle missing data gracefully
   - Consider different timeframes

3. **Reallocation Timing**
   - Avoid excessive rebalancing
   - Consider transaction costs
   - Implement minimum change thresholds

## 🎯 Success Criteria

Step 5 is complete when:
1. ✅ Multiple risk containers isolated
2. ✅ Portfolio constraints working
3. ✅ Correlation monitoring active
4. ✅ Dynamic allocation functional
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 6: Multiple Classifiers](step-06-multiple-classifiers.md)

## 📚 Additional Resources

- [Risk Parity Theory](../references/risk-parity.md)
- [Kelly Criterion](../references/kelly-criterion.md)
- [Correlation Risk](../references/correlation-risk.md)