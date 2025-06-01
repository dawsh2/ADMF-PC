# Step 8: Signal Replay

**Status**: Signal Capture & Replay Step
**Complexity**: High
**Prerequisites**: [Step 7: Signal Capture](step-07-signal-capture.md) completed
**Architecture Ref**: [EVENT-DRIVEN-ARCHITECTURE.md](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md#event-replay)

## 🎯 Objective

Implement signal replay system for fast optimization:
- Replay captured signals with different parameters
- Skip strategy calculation phase entirely
- Test different risk settings instantly
- Enable rapid parameter sensitivity analysis
- Support what-if scenario testing

## 📋 Required Reading

Before starting:
1. [Signal Capture](step-07-signal-capture.md)
2. [Event Replay Patterns](../../patterns/event-replay.md)
3. [Optimization Workflows](../../legacy/MULTIPHASE_OPTIMIZATION.md)

## 🏗️ Implementation Tasks

### 1. Signal Replay Engine

```python
# src/execution/signal_replay_engine.py
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class ReplayConfig:
    """Configuration for signal replay"""
    # Risk parameters to modify
    position_sizing_method: str = 'fixed'
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.06
    
    # Execution parameters
    slippage_model: str = 'linear'
    commission_model: str = 'per_share'
    fill_ratio: float = 1.0
    
    # Filters
    signal_filters: Optional[Dict[str, Any]] = None
    time_filters: Optional[Dict[str, Any]] = None
    
    # Performance
    use_vectorized: bool = True
    chunk_size: int = 10000

class SignalReplayEngine:
    """
    Replays captured signals with modified parameters.
    Achieves 10x+ speedup by skipping strategy calculations.
    """
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self.loaded_signals: Optional[pd.DataFrame] = None
        self.replay_results: List[ReplayResult] = []
        self.logger = ComponentLogger("SignalReplay", "replay")
        
        # Performance tracking
        self.replay_count = 0
        self.total_replay_time = 0.0
    
    def load_signals(self, start_date: datetime, end_date: datetime,
                    filters: Optional[Dict] = None) -> int:
        """Load signals for replay"""
        self.logger.info(
            f"Loading signals from {start_date} to {end_date}"
        )
        
        # Load from storage
        self.loaded_signals = self.storage.load_signals({
            'timestamp': {'>=': start_date, '<=': end_date},
            **(filters or {})
        })
        
        # Parse JSON fields if needed
        if 'portfolio_state_json' in self.loaded_signals.columns:
            self.loaded_signals['portfolio_state'] = \
                self.loaded_signals['portfolio_state_json'].apply(json.loads)
        
        signal_count = len(self.loaded_signals)
        self.logger.info(f"Loaded {signal_count} signals for replay")
        
        return signal_count
    
    def replay(self, config: ReplayConfig, 
              progress_callback: Optional[Callable] = None) -> ReplayResult:
        """Replay signals with new parameters"""
        if self.loaded_signals is None or self.loaded_signals.empty:
            raise ValueError("No signals loaded for replay")
        
        start_time = time.time()
        
        # Apply filters if specified
        signals_to_replay = self._apply_filters(
            self.loaded_signals, config.signal_filters
        )
        
        self.logger.info(
            f"Starting replay of {len(signals_to_replay)} signals"
        )
        
        # Initialize replay state
        replay_state = ReplayState(
            initial_capital=self._get_initial_capital(),
            config=config
        )
        
        # Choose replay method
        if config.use_vectorized and self._can_vectorize(signals_to_replay):
            results = self._vectorized_replay(signals_to_replay, replay_state, config)
        else:
            results = self._sequential_replay(signals_to_replay, replay_state, 
                                            config, progress_callback)
        
        # Calculate performance metrics
        replay_time = time.time() - start_time
        final_result = self._calculate_final_metrics(results, replay_state, replay_time)
        
        # Store result
        self.replay_results.append(final_result)
        self.replay_count += 1
        self.total_replay_time += replay_time
        
        self.logger.info(
            f"Replay complete in {replay_time:.2f}s. "
            f"Final return: {final_result.total_return:.2%}"
        )
        
        return final_result
    
    def _sequential_replay(self, signals: pd.DataFrame, state: ReplayState,
                         config: ReplayConfig, 
                         progress_callback: Optional[Callable]) -> List[Trade]:
        """Sequential replay for complex logic"""
        trades = []
        total_signals = len(signals)
        
        for idx, (_, signal_row) in enumerate(signals.iterrows()):
            # Reconstruct signal
            signal = self._reconstruct_signal(signal_row)
            
            # Apply new risk parameters
            position_size = self._calculate_position_size(
                signal, state, config
            )
            
            if position_size > 0:
                # Create order
                order = self._create_order(
                    signal, position_size, state, config
                )
                
                # Simulate execution
                fill = self._simulate_execution(
                    order, signal_row, config
                )
                
                if fill:
                    # Update state
                    trade = self._process_fill(fill, state)
                    trades.append(trade)
                    
                    # Update portfolio state
                    state.update_from_trade(trade)
            
            # Progress callback
            if progress_callback and idx % 100 == 0:
                progress = (idx + 1) / total_signals
                progress_callback(progress, state)
        
        return trades
    
    def _vectorized_replay(self, signals: pd.DataFrame, state: ReplayState,
                         config: ReplayConfig) -> List[Trade]:
        """Vectorized replay for maximum performance"""
        # Pre-calculate position sizes for all signals
        signals['position_size'] = self._vectorized_position_sizing(
            signals, state.initial_capital, config
        )
        
        # Filter viable trades
        viable_signals = signals[signals['position_size'] > 0]
        
        # Vectorized execution simulation
        trades_df = self._vectorized_execution(
            viable_signals, config
        )
        
        # Convert to Trade objects
        trades = self._dataframe_to_trades(trades_df)
        
        return trades
    
    def _calculate_position_size(self, signal: CapturedSignal, 
                               state: ReplayState,
                               config: ReplayConfig) -> float:
        """Calculate position size with new risk parameters"""
        if config.position_sizing_method == 'fixed':
            # Fixed percentage of capital
            position_value = state.current_capital * config.max_position_size
            position_size = position_value / signal.market_data['close']
            
        elif config.position_sizing_method == 'risk_based':
            # Kelly-inspired sizing
            risk_amount = state.current_capital * config.risk_per_trade
            stop_distance = self._calculate_stop_distance(signal)
            position_size = risk_amount / stop_distance
            
        elif config.position_sizing_method == 'volatility_adjusted':
            # Size inversely proportional to volatility
            volatility = signal.indicators.get('ATR', signal.market_data['close'] * 0.02)
            base_size = state.current_capital * config.max_position_size
            position_size = base_size / (signal.market_data['close'] * (1 + volatility))
            
        else:
            raise ValueError(f"Unknown sizing method: {config.position_sizing_method}")
        
        # Apply position limits
        max_shares = (state.current_capital * config.max_position_size) / \
                    signal.market_data['close']
        position_size = min(position_size, max_shares)
        
        # Check portfolio risk
        if not self._check_portfolio_risk(position_size, signal, state, config):
            return 0.0
        
        return position_size
    
    def _simulate_execution(self, order: Order, signal_data: pd.Series,
                          config: ReplayConfig) -> Optional[Fill]:
        """Simulate order execution with slippage and commission"""
        # Base fill price
        if order.order_type == OrderType.MARKET:
            base_price = signal_data['market_ask'] if order.direction == Direction.BUY \
                       else signal_data['market_bid']
        else:
            base_price = order.limit_price
        
        # Apply slippage
        slippage = self._calculate_slippage(
            order.quantity, signal_data, config
        )
        fill_price = base_price * (1 + slippage) if order.direction == Direction.BUY \
                   else base_price * (1 - slippage)
        
        # Apply fill ratio
        fill_quantity = order.quantity * config.fill_ratio
        
        if fill_quantity == 0:
            return None
        
        # Calculate commission
        commission = self._calculate_commission(
            fill_quantity, fill_price, config
        )
        
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            direction=order.direction,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            timestamp=signal_data['timestamp'],
            execution_id=f"replay_{order.order_id}"
        )
```

### 2. Replay State Management

```python
# src/execution/replay_state.py
class ReplayState:
    """
    Maintains portfolio state during replay.
    Optimized for fast updates.
    """
    
    def __init__(self, initial_capital: float, config: ReplayConfig):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        
        # Performance tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
        
        # Risk tracking
        self.current_exposure = 0.0
        self.max_exposure_reached = 0.0
        self.risk_violations = 0
    
    def update_from_trade(self, trade: Trade) -> None:
        """Update state from executed trade"""
        # Update position
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = Position(trade.symbol)
        
        position = self.positions[trade.symbol]
        position.update_from_trade(trade)
        
        # Update capital
        trade_cost = trade.quantity * trade.price * trade.direction.value
        self.current_capital -= (trade_cost + trade.commission)
        
        # Update exposure
        self._update_exposure()
        
        # Track trade
        self.trades.append(trade)
        
        # Update equity curve
        current_equity = self.calculate_total_equity()
        self.equity_curve.append(current_equity)
        
        # Update drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        else:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def calculate_total_equity(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value"""
        # Cash component
        total = self.current_capital
        
        # Position values
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # Use provided price or last trade price
                price = market_prices.get(symbol) if market_prices else position.last_price
                total += position.quantity * price
        
        return total
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance metrics"""
        current_equity = self.calculate_total_equity()
        total_return = (current_equity - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 2:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_return': total_return,
            'current_equity': current_equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'avg_trade_pnl': self.realized_pnl / len(self.trades) if self.trades else 0,
            'max_exposure': self.max_exposure_reached,
            'risk_violations': self.risk_violations
        }
```

### 3. Optimization Interface

```python
# src/optimization/replay_optimizer.py
class ReplayOptimizer:
    """
    Optimizes parameters using signal replay.
    Dramatically faster than traditional optimization.
    """
    
    def __init__(self, replay_engine: SignalReplayEngine):
        self.replay_engine = replay_engine
        self.optimization_results: List[OptimizationResult] = []
        self.logger = ComponentLogger("ReplayOptimizer", "optimization")
    
    def optimize(self, param_grid: Dict[str, List[Any]], 
                objective: str = 'sharpe_ratio',
                constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize parameters using grid search with replay"""
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        self.logger.info(
            f"Starting optimization with {total_combinations} parameter combinations"
        )
        
        # Track best result
        best_result = None
        best_objective_value = float('-inf')
        all_results = []
        
        # Test each combination
        for idx, params in enumerate(param_combinations):
            # Create replay config
            replay_config = ReplayConfig(**params)
            
            # Check constraints
            if constraints and not self._check_constraints(params, constraints):
                continue
            
            # Run replay
            try:
                result = self.replay_engine.replay(replay_config)
                
                # Extract objective value
                objective_value = self._get_objective_value(result, objective)
                
                # Track result
                all_results.append({
                    'params': params,
                    'result': result,
                    'objective_value': objective_value
                })
                
                # Update best
                if objective_value > best_objective_value:
                    best_objective_value = objective_value
                    best_result = result
                    best_params = params
                
                # Progress update
                if (idx + 1) % 10 == 0:
                    self.logger.info(
                        f"Progress: {idx + 1}/{total_combinations} "
                        f"Best {objective}: {best_objective_value:.4f}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to replay with params {params}: {e}")
                continue
        
        # Create optimization result
        optimization_result = OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_objective_value=best_objective_value,
            all_results=all_results,
            param_grid=param_grid,
            objective=objective,
            total_combinations_tested=len(all_results)
        )
        
        self.optimization_results.append(optimization_result)
        
        return optimization_result
    
    def sensitivity_analysis(self, base_params: Dict[str, Any],
                           param_ranges: Dict[str, Tuple[float, float]],
                           n_points: int = 20) -> SensitivityResult:
        """Analyze parameter sensitivity around base case"""
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            # Create parameter sweep
            param_values = np.linspace(min_val, max_val, n_points)
            results = []
            
            for value in param_values:
                # Modify single parameter
                test_params = base_params.copy()
                test_params[param_name] = value
                
                # Run replay
                config = ReplayConfig(**test_params)
                result = self.replay_engine.replay(config)
                
                results.append({
                    'value': value,
                    'sharpe': result.sharpe_ratio,
                    'return': result.total_return,
                    'drawdown': result.max_drawdown
                })
            
            sensitivity_results[param_name] = pd.DataFrame(results)
        
        return SensitivityResult(
            base_params=base_params,
            sensitivity_data=sensitivity_results,
            optimal_ranges=self._identify_optimal_ranges(sensitivity_results)
        )
```

### 4. Replay Validation

```python
# src/execution/replay_validation.py
class ReplayValidator:
    """
    Validates that replay produces identical results to original backtest.
    Critical for ensuring replay accuracy.
    """
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.logger = ComponentLogger("ReplayValidator", "validation")
    
    def validate_replay_accuracy(self, original_results: BacktestResults,
                               replay_results: ReplayResult,
                               tolerance: float = 1e-10) -> ValidationResult:
        """Validate replay matches original results exactly"""
        discrepancies = []
        
        # Check trade count
        if original_results.total_trades != replay_results.total_trades:
            discrepancies.append(
                f"Trade count mismatch: {original_results.total_trades} vs "
                f"{replay_results.total_trades}"
            )
        
        # Check final equity
        equity_diff = abs(original_results.final_equity - replay_results.final_equity)
        if equity_diff > tolerance:
            discrepancies.append(
                f"Final equity mismatch: {original_results.final_equity} vs "
                f"{replay_results.final_equity} (diff: {equity_diff})"
            )
        
        # Check key metrics
        metrics_to_check = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in metrics_to_check:
            orig_value = getattr(original_results, metric)
            replay_value = getattr(replay_results, metric)
            
            if abs(orig_value - replay_value) > tolerance:
                discrepancies.append(
                    f"{metric} mismatch: {orig_value} vs {replay_value}"
                )
        
        # Check individual trades if available
        if hasattr(original_results, 'trades') and hasattr(replay_results, 'trades'):
            trade_discrepancies = self._validate_trades(
                original_results.trades, 
                replay_results.trades,
                tolerance
            )
            discrepancies.extend(trade_discrepancies)
        
        # Create validation result
        is_valid = len(discrepancies) == 0
        
        validation_result = ValidationResult(
            is_valid=is_valid,
            discrepancies=discrepancies,
            original_metrics=original_results.get_metrics(),
            replay_metrics=replay_results.get_metrics(),
            tolerance_used=tolerance
        )
        
        self.validation_results.append(validation_result)
        
        if is_valid:
            self.logger.info("✅ Replay validation passed")
        else:
            self.logger.error(
                f"❌ Replay validation failed with {len(discrepancies)} discrepancies"
            )
            for discrepancy in discrepancies[:5]:  # Show first 5
                self.logger.error(f"  - {discrepancy}")
        
        return validation_result
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step8_signal_replay.py`:

```python
class TestSignalReplay:
    """Test signal replay functionality"""
    
    def test_position_sizing_methods(self):
        """Test different position sizing calculations"""
        # Create test signal
        signal = create_test_captured_signal(
            market_price=100.0,
            indicators={'ATR': 2.0}
        )
        
        state = ReplayState(initial_capital=100000, config=ReplayConfig())
        
        # Test fixed sizing
        config_fixed = ReplayConfig(
            position_sizing_method='fixed',
            max_position_size=0.1
        )
        size_fixed = calculate_position_size(signal, state, config_fixed)
        assert size_fixed == 100  # 10% of 100k at $100/share
        
        # Test risk-based sizing
        config_risk = ReplayConfig(
            position_sizing_method='risk_based',
            risk_per_trade=0.02
        )
        size_risk = calculate_position_size(signal, state, config_risk)
        assert size_risk > 0
    
    def test_replay_state_updates(self):
        """Test state management during replay"""
        state = ReplayState(100000, ReplayConfig())
        
        # Create and process trade
        trade = Trade(
            symbol='AAPL',
            direction=Direction.BUY,
            quantity=100,
            price=150.0,
            commission=1.0,
            timestamp=datetime.now()
        )
        
        state.update_from_trade(trade)
        
        # Verify state
        assert 'AAPL' in state.positions
        assert state.positions['AAPL'].quantity == 100
        assert state.current_capital < 100000 - 15000  # Bought 100 @ 150
        assert len(state.equity_curve) == 2
```

### Integration Tests

Create `tests/integration/test_step8_replay_integration.py`:

```python
def test_replay_matches_original():
    """Test replay produces identical results to original"""
    # Run original backtest with capture
    capture_engine = SignalCaptureEngine(test_capture_config())
    original_strategy = CaptureAwareStrategy(
        MomentumStrategy(), capture_engine
    )
    
    original_results = run_backtest(
        strategy=original_strategy,
        data=test_data,
        config=original_config
    )
    
    # Flush captured signals
    capture_engine.flush()
    
    # Create replay engine
    replay_engine = SignalReplayEngine(capture_engine.storage_backend)
    replay_engine.load_signals(
        start_date=test_data.index[0],
        end_date=test_data.index[-1]
    )
    
    # Replay with same parameters
    replay_config = ReplayConfig(
        position_sizing_method=original_config['position_sizing_method'],
        max_position_size=original_config['max_position_size']
    )
    
    replay_results = replay_engine.replay(replay_config)
    
    # Validate exact match
    validator = ReplayValidator()
    validation = validator.validate_replay_accuracy(
        original_results, replay_results
    )
    
    assert validation.is_valid
    assert len(validation.discrepancies) == 0

def test_replay_optimization_performance():
    """Test replay optimization is faster than traditional"""
    # Setup captured signals
    setup_test_signals(n_signals=1000)
    
    # Traditional optimization timing
    start_time = time.time()
    traditional_results = traditional_optimize(
        param_grid={'max_position_size': [0.05, 0.1, 0.15, 0.2]},
        n_iterations=10
    )
    traditional_time = time.time() - start_time
    
    # Replay optimization timing
    replay_engine = SignalReplayEngine(test_storage())
    replay_engine.load_signals(datetime(2024, 1, 1), datetime(2024, 12, 31))
    
    optimizer = ReplayOptimizer(replay_engine)
    
    start_time = time.time()
    replay_results = optimizer.optimize(
        param_grid={'max_position_size': [0.05, 0.1, 0.15, 0.2]},
        objective='sharpe_ratio'
    )
    replay_time = time.time() - start_time
    
    # Verify speedup
    speedup = traditional_time / replay_time
    assert speedup > 5  # At least 5x faster
    
    # Verify same optimal parameters found
    assert abs(
        traditional_results.best_params['max_position_size'] - 
        replay_results.best_params['max_position_size']
    ) < 0.01
```

### System Tests

Create `tests/system/test_step8_replay_system.py`:

```python
def test_sensitivity_analysis_workflow():
    """Test complete sensitivity analysis workflow"""
    # Run initial backtest with signal capture
    system = create_system_with_capture()
    initial_results = system.run_backtest(
        data=load_test_data(),
        capture=True
    )
    
    # Setup replay engine
    replay_engine = SignalReplayEngine(system.capture_engine.storage_backend)
    replay_engine.load_signals(
        datetime(2024, 1, 1), datetime(2024, 12, 31)
    )
    
    # Run sensitivity analysis
    optimizer = ReplayOptimizer(replay_engine)
    
    base_params = {
        'max_position_size': 0.1,
        'risk_per_trade': 0.02,
        'max_portfolio_risk': 0.06
    }
    
    param_ranges = {
        'max_position_size': (0.05, 0.20),
        'risk_per_trade': (0.01, 0.04),
        'max_portfolio_risk': (0.04, 0.10)
    }
    
    sensitivity_result = optimizer.sensitivity_analysis(
        base_params, param_ranges, n_points=20
    )
    
    # Verify sensitivity analysis
    assert 'max_position_size' in sensitivity_result.sensitivity_data
    assert len(sensitivity_result.sensitivity_data['max_position_size']) == 20
    
    # Check optimal ranges identified
    assert sensitivity_result.optimal_ranges['max_position_size']['min'] > 0.05
    assert sensitivity_result.optimal_ranges['max_position_size']['max'] < 0.20

def test_multi_strategy_replay_optimization():
    """Test replay optimization with multiple strategies"""
    # Capture signals from multiple strategies
    strategies = ['momentum', 'mean_reversion', 'trend_following']
    
    for strategy_name in strategies:
        run_strategy_with_capture(strategy_name, test_data)
    
    # Load all signals
    replay_engine = SignalReplayEngine(global_storage())
    signal_count = replay_engine.load_signals(
        datetime(2024, 1, 1), datetime(2024, 12, 31)
    )
    
    assert signal_count > 1000
    
    # Optimize risk parameters for all strategies combined
    optimizer = ReplayOptimizer(replay_engine)
    
    result = optimizer.optimize(
        param_grid={
            'max_position_size': np.linspace(0.05, 0.15, 10),
            'max_portfolio_risk': np.linspace(0.04, 0.08, 5)
        },
        objective='sharpe_ratio',
        constraints={'max_drawdown': {'max': 0.15}}
    )
    
    # Verify optimization found improvement
    assert result.best_objective_value > 1.0  # Sharpe > 1
    assert result.total_combinations_tested == 50
    assert result.best_result.max_drawdown <= 0.15
```

## ✅ Validation Checklist

### Replay Accuracy
- [ ] Replay matches original results exactly
- [ ] All trades reproduced correctly
- [ ] Final equity matches to high precision
- [ ] Performance metrics identical

### Optimization Performance
- [ ] 10x+ speedup achieved
- [ ] Parameter grid search efficient
- [ ] Memory usage reasonable
- [ ] Results reproducible

### Feature Completeness
- [ ] All position sizing methods work
- [ ] Slippage models implemented
- [ ] Commission calculation accurate
- [ ] Constraints enforced

## 📊 Memory & Performance

### Performance Optimization
```python
class VectorizedReplayEngine:
    """Ultra-fast vectorized replay implementation"""
    
    def vectorized_backtest(self, signals_df: pd.DataFrame, 
                          config: ReplayConfig) -> pd.DataFrame:
        """Fully vectorized replay for maximum speed"""
        # Pre-calculate all position sizes
        signals_df['position_value'] = (
            signals_df['capital_available'] * config.max_position_size
        )
        signals_df['position_size'] = (
            signals_df['position_value'] / signals_df['market_close']
        )
        
        # Vectorized slippage
        signals_df['slippage'] = (
            signals_df['position_size'] * signals_df['spread'] * 0.5
        )
        
        # Vectorized fills
        signals_df['fill_price'] = np.where(
            signals_df['direction'] == 1,  # Buy
            signals_df['market_ask'] + signals_df['slippage'],
            signals_df['market_bid'] - signals_df['slippage']
        )
        
        # Calculate P&L
        signals_df['trade_pnl'] = self._vectorized_pnl(signals_df)
        
        return signals_df
```

### Memory Management
- Signal chunking for large datasets
- Streaming replay for unlimited signals
- Efficient storage formats
- Result compression

## 🐛 Common Issues

1. **Replay Drift**
   - Ensure all context captured
   - Validate before optimization
   - Check floating point precision

2. **Performance Bottlenecks**
   - Profile replay execution
   - Use vectorization where possible
   - Optimize data access patterns

3. **Parameter Explosion**
   - Use smart grid search
   - Implement early stopping
   - Consider Bayesian optimization

## 🎯 Success Criteria

Step 8 is complete when:
1. ✅ Replay engine fully functional
2. ✅ Exact reproduction validated
3. ✅ 10x performance improvement achieved
4. ✅ Optimization interface working
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 8.5: Statistical Validation](step-08.5-monte-carlo.md)

## 📚 Additional Resources

- [Optimization Theory](../references/optimization-theory.md)
- [Backtesting Best Practices](../references/backtesting-practices.md)
- [Parameter Sensitivity Analysis](../references/sensitivity-analysis.md)