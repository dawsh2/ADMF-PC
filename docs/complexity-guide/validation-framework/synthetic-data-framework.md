# Synthetic Data Validation Framework

## Overview

The Synthetic Data Validation Framework enables deterministic testing with pre-computed expected results. This ensures that your implementation is correct by comparing actual outputs against known, hand-calculated results.

## Purpose

- **Deterministic Testing**: Remove randomness from tests
- **Exact Validation**: Compare results to pre-calculated expectations
- **Debugging Aid**: Easily identify where logic differs from expectations
- **Confidence Building**: Know your implementation is mathematically correct

## Core Components

### SyntheticTestFramework Class

```python
class SyntheticTestFramework:
    """Framework for testing with synthetic data and contrived rules"""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.logger = ComponentLogger("synthetic_test", container_id)
        self.expected_results = {}
        self.actual_results = {}
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create deterministic market data for testing"""
        # Simple deterministic price pattern
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Contrived pattern: price oscillates between 100-110 with trend
        base_prices = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 5
        trend = np.linspace(0, 10, 100)  # 10% trend over period
        prices = base_prices + trend
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.002, 
            'low': prices * 0.998,
            'close': prices,
            'volume': np.full(100, 1000000)
        })
    
    def create_contrived_strategy(self) -> Dict[str, Any]:
        """Create strategy with deterministic, pre-computable signals"""
        return {
            "type": "synthetic_sma_cross",
            "rules": {
                "sma_short": 5,  # 5-day SMA
                "sma_long": 10,  # 10-day SMA
                "signal_strength": 1.0,  # Always full strength
                "entry_rule": "short_crosses_above_long",
                "exit_rule": "short_crosses_below_long"
            }
        }
```

## Usage Pattern

### 1. Create Synthetic Data
```python
synthetic = SyntheticTestFramework("test_container")
test_data = synthetic.create_synthetic_data()
```

### 2. Define Contrived Strategy
```python
strategy_config = synthetic.create_contrived_strategy()
```

### 3. Pre-compute Expected Results
```python
expected = synthetic.compute_expected_results(test_data, strategy_config)
```

### 4. Run Actual Implementation
```python
actual_results = run_backtest(test_data, strategy_config)
```

### 5. Validate Results Match
```python
validation_passed = synthetic.validate_results(expected, actual_results)
assert validation_passed, "Results don't match expected values!"
```

## Pre-computation Example

### Simple Moving Average Crossover
```python
def compute_expected_sma_crossover_trades(data: pd.DataFrame, 
                                         short_period: int = 5, 
                                         long_period: int = 10):
    """Pre-compute expected trades for SMA crossover strategy"""
    
    # Calculate SMAs
    data['sma_short'] = data['close'].rolling(short_period).mean()
    data['sma_long'] = data['close'].rolling(long_period).mean()
    
    # Determine signals
    data['position'] = 0
    data.loc[data['sma_short'] > data['sma_long'], 'position'] = 1
    data.loc[data['sma_short'] < data['sma_long'], 'position'] = -1
    
    # Find signal changes (trades)
    data['signal_change'] = data['position'].diff()
    
    trades = []
    for idx, row in data[data['signal_change'] != 0].iterrows():
        if row['signal_change'] == 2:  # -1 to 1 (Buy)
            trades.append({
                'timestamp': row['timestamp'],
                'action': 'BUY',
                'price': row['close'],
                'quantity': 100
            })
        elif row['signal_change'] == -2:  # 1 to -1 (Sell)
            trades.append({
                'timestamp': row['timestamp'],
                'action': 'SELL',
                'price': row['close'],
                'quantity': 100
            })
    
    return trades
```

## Validation Functions

### Trade Validation
```python
def validate_trades(expected_trades: List[Dict], actual_trades: List[Dict]) -> bool:
    """Validate that actual trades match expected trades exactly"""
    
    if len(expected_trades) != len(actual_trades):
        print(f"Trade count mismatch: expected {len(expected_trades)}, got {len(actual_trades)}")
        return False
    
    for i, (exp, act) in enumerate(zip(expected_trades, actual_trades)):
        if exp['action'] != act['action']:
            print(f"Trade {i}: action mismatch - expected {exp['action']}, got {act['action']}")
            return False
        
        if abs(exp['price'] - act['price']) > 0.01:
            print(f"Trade {i}: price mismatch - expected {exp['price']}, got {act['price']}")
            return False
        
        if exp['quantity'] != act['quantity']:
            print(f"Trade {i}: quantity mismatch - expected {exp['quantity']}, got {act['quantity']}")
            return False
    
    return True
```

### Portfolio Validation
```python
def validate_portfolio_evolution(expected_states: List[Dict], 
                               actual_states: List[Dict]) -> bool:
    """Validate portfolio state evolution matches expectations"""
    
    for i, (exp, act) in enumerate(zip(expected_states, actual_states)):
        cash_diff = abs(exp['cash'] - act['cash'])
        if cash_diff > 0.01:
            print(f"State {i}: cash mismatch - expected {exp['cash']}, got {act['cash']}")
            return False
        
        if exp['shares'] != act['shares']:
            print(f"State {i}: shares mismatch - expected {exp['shares']}, got {act['shares']}")
            return False
        
        value_diff = abs(exp['total_value'] - act['total_value'])
        if value_diff > 0.01:
            print(f"State {i}: value mismatch - expected {exp['total_value']}, got {act['total_value']}")
            return False
    
    return True
```

## Test Scenarios

### 1. Basic Signal Generation
```python
def test_basic_signal_generation():
    """Test that signals are generated at correct times"""
    synthetic = SyntheticTestFramework("test_basic")
    
    # Create simple price series: [100, 105, 110, 105, 100]
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5),
        'close': [100, 105, 110, 105, 100]
    })
    
    # With SMA(2), expect:
    # Day 2: SMA = 102.5, Price = 105 → BUY signal
    # Day 4: SMA = 107.5, Price = 105 → SELL signal
    
    expected_signals = [
        {'day': 2, 'signal': 'BUY', 'sma': 102.5},
        {'day': 4, 'signal': 'SELL', 'sma': 107.5}
    ]
    
    # Run test and validate
    actual_signals = strategy.generate_signals(data)
    assert len(actual_signals) == len(expected_signals)
```

### 2. Risk Management Validation
```python
def test_position_sizing():
    """Test that position sizing follows risk rules"""
    
    # Given: $10,000 capital, 2% risk per trade
    # Price: $100, Stop loss: $95 (5% risk)
    # Expected position size: $10,000 * 0.02 / ($100 - $95) = 40 shares
    
    risk_manager = RiskManager(capital=10000, risk_percent=0.02)
    position_size = risk_manager.calculate_position_size(
        entry_price=100,
        stop_loss=95
    )
    
    assert position_size == 40, f"Expected 40 shares, got {position_size}"
```

### 3. Multi-Asset Coordination
```python
def test_multi_asset_signals():
    """Test signal coordination across multiple assets"""
    
    # Create correlated synthetic data
    spy_prices = [100, 102, 104, 103, 101]
    qqq_prices = [200, 205, 210, 208, 204]  # 2x SPY with same pattern
    
    # Expect correlation-based signals
    expected_divergence_signal = {
        'timestamp': '2024-01-04',
        'signal_type': 'DIVERGENCE',
        'assets': ['SPY', 'QQQ'],
        'action': 'PAIR_TRADE'
    }
    
    # Validate detection
    actual_signals = correlation_strategy.analyze(spy_prices, qqq_prices)
    assert divergence_detected(actual_signals, expected_divergence_signal)
```

## Integration with Complexity Steps

Each step must include synthetic validation:

```python
# Step 1: Core Pipeline Test
def test_step1_with_synthetic_data():
    synthetic = SyntheticTestFramework("step1_container")
    
    # Generate deterministic data
    test_data = synthetic.create_synthetic_data()
    contrived_strategy = synthetic.create_contrived_strategy()
    
    # Pre-compute expected results
    expected = synthetic.compute_expected_results(test_data, contrived_strategy)
    
    # Run actual backtest
    backtest_result = run_step1_backtest(test_data, contrived_strategy)
    
    # Validate results match exactly
    validation_passed = synthetic.validate_results(expected, backtest_result)
    
    assert validation_passed, "Synthetic validation failed"
```

## Benefits

1. **Confidence**: Know your implementation is correct
2. **Debugging**: Pinpoint exactly where results diverge
3. **Documentation**: Expected results serve as specification
4. **Regression Testing**: Ensure changes don't break correctness
5. **Onboarding**: New developers can understand expected behavior

## Next Steps

1. Implement `SyntheticTestFramework` class
2. Create test data generators for each strategy type
3. Build expected result calculators
4. Integrate with each complexity step
5. Add to CI/CD pipeline