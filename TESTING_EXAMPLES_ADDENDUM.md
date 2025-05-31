# Testing Examples Addendum for COMPLEXITY_CHECKLIST.MD

Add these specific examples after the "Three-Tier Testing Strategy" section:

---

## Concrete Testing Examples for Each Tier

### Unit Test Example: Testing Indicator Calculation

```python
# tests/unit/strategy/test_indicators.py
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.strategy.components.indicators import SMAIndicator, RSIIndicator

class TestSMAIndicator:
    """Unit tests for Simple Moving Average indicator"""
    
    @pytest.fixture
    def synthetic_prices(self):
        """Create deterministic price series for testing"""
        # Simple ascending price series for easy validation
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = np.arange(100, 120, 1.0)  # 100, 101, 102, ..., 119
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
    
    def test_sma_calculation_correctness(self, synthetic_prices):
        """Test SMA calculates correct values"""
        sma = SMAIndicator(period=5)
        
        # Expected values for 5-period SMA
        # First 4 values should be None (insufficient data)
        # 5th value: (100+101+102+103+104)/5 = 102.0
        # 6th value: (101+102+103+104+105)/5 = 103.0
        expected_values = [None, None, None, None, 102.0, 103.0, 104.0, 105.0]
        
        actual_values = []
        for idx, row in synthetic_prices.head(8).iterrows():
            value = sma.calculate(row['close'])
            actual_values.append(value)
        
        # Validate each value
        for i, (expected, actual) in enumerate(zip(expected_values, actual_values)):
            if expected is None:
                assert actual is None, f"Index {i}: Expected None, got {actual}"
            else:
                assert abs(expected - actual) < 1e-10, f"Index {i}: Expected {expected}, got {actual}"
    
    def test_sma_state_management(self, synthetic_prices):
        """Test SMA maintains correct internal state"""
        sma = SMAIndicator(period=3)
        
        # Feed first 3 prices
        for idx in range(3):
            sma.calculate(synthetic_prices.iloc[idx]['close'])
        
        # Check internal state
        assert len(sma._price_buffer) == 3
        assert sma._price_buffer == [100.0, 101.0, 102.0]
        
        # Add one more price
        sma.calculate(103.0)
        
        # Buffer should maintain size 3 (FIFO)
        assert len(sma._price_buffer) == 3
        assert sma._price_buffer == [101.0, 102.0, 103.0]
    
    def test_sma_reset_functionality(self, synthetic_prices):
        """Test SMA can be reset to initial state"""
        sma = SMAIndicator(period=5)
        
        # Calculate some values
        for idx in range(10):
            sma.calculate(synthetic_prices.iloc[idx]['close'])
        
        # Reset
        sma.reset()
        
        # Verify clean state
        assert len(sma._price_buffer) == 0
        assert sma.calculate(100.0) is None  # Should need 5 values again
```

### Integration Test Example: Testing Signal to Order Flow

```python
# tests/integration/test_signal_to_order_flow.py
import pytest
import time
from datetime import datetime
from src.core.events.event_bus import EventBus
from src.core.containers.factory import create_container
from src.strategy.protocols import Signal
from src.risk.protocols import Order
from src.core.events.enhanced_isolation import get_enhanced_isolation_manager

class TestSignalToOrderFlow:
    """Integration tests for signal processing through risk management"""
    
    @pytest.fixture
    def test_environment(self):
        """Create isolated test environment with containers"""
        isolation_manager = get_enhanced_isolation_manager()
        
        # Create containers
        strategy_container = create_container("strategy", "test_strategy_001")
        risk_container = create_container("risk", "test_risk_001")
        execution_container = create_container("execution", "test_execution_001")
        
        # Wire up event flows
        strategy_container.connect_to(risk_container, ["SIGNAL_EVENT"])
        risk_container.connect_to(execution_container, ["ORDER_EVENT"])
        
        return {
            'isolation_manager': isolation_manager,
            'strategy': strategy_container,
            'risk': risk_container,
            'execution': execution_container,
            'captured_events': []
        }
    
    def test_signal_transforms_to_order_with_position_sizing(self, test_environment):
        """Test that signals are properly transformed to orders with position sizing"""
        env = test_environment
        
        # Setup event capture
        def capture_order(event):
            if event.type == "ORDER_EVENT":
                env['captured_events'].append(event)
        
        env['execution'].subscribe("ORDER_EVENT", capture_order)
        
        # Create test signal
        test_signal = Signal(
            timestamp=datetime.now(),
            symbol="TEST",
            direction="BUY",
            strength=0.8,  # 80% signal strength
            strategy_id="momentum",
            metadata={"reason": "test_signal"}
        )
        
        # Configure risk container with test portfolio
        env['risk'].update_portfolio_state({
            'cash': 10000,
            'positions': {},
            'total_value': 10000
        })
        
        # Emit signal from strategy container
        env['strategy'].emit("SIGNAL_EVENT", test_signal)
        
        # Allow async processing
        time.sleep(0.1)
        
        # Validate order was created
        assert len(env['captured_events']) == 1
        order_event = env['captured_events'][0]
        order = order_event.payload
        
        # Validate order properties
        assert isinstance(order, Order)
        assert order.symbol == "TEST"
        assert order.side == "BUY"
        assert order.quantity == 80  # 0.8 * 100 shares (position sizing)
        assert order.order_type == "MARKET"
        assert order.source_signal_id == test_signal.id
        
        # Validate container isolation
        assert order_event.container_id == "test_risk_001"
        assert not env['isolation_manager'].has_violations()
    
    def test_risk_limits_prevent_excessive_orders(self, test_environment):
        """Test that risk limits properly constrain order generation"""
        env = test_environment
        
        # Configure risk container with limited capital
        env['risk'].update_portfolio_state({
            'cash': 1000,  # Limited cash
            'positions': {},
            'total_value': 1000
        })
        
        # Set risk limits
        env['risk'].set_risk_limits({
            'max_position_size': 0.2,  # Max 20% per position
            'max_total_exposure': 0.8  # Max 80% total exposure
        })
        
        captured_orders = []
        def capture_order(event):
            if event.type == "ORDER_EVENT":
                captured_orders.append(event.payload)
        
        env['execution'].subscribe("ORDER_EVENT", capture_order)
        
        # Create large signal
        large_signal = Signal(
            timestamp=datetime.now(),
            symbol="TEST",
            direction="BUY",
            strength=1.0,  # Full strength
            strategy_id="momentum"
        )
        
        # Emit signal
        env['strategy'].emit("SIGNAL_EVENT", large_signal)
        time.sleep(0.1)
        
        # Validate order was constrained by risk limits
        assert len(captured_orders) == 1
        order = captured_orders[0]
        
        # With $1000 capital and 20% max position, max order is $200
        # Assuming $10/share, max quantity is 20
        assert order.quantity <= 20
        assert order.metadata.get('risk_constrained') == True
```

### System Test Example: Complete Backtest Workflow

```python
# tests/system/test_complete_backtest_workflow.py
import pytest
import json
from pathlib import Path
from src.core.coordinator.yaml_coordinator import YAMLCoordinator
from tests.fixtures.synthetic_data import create_trending_market_data

class TestCompleteBacktestWorkflow:
    """System tests for complete backtest execution with deterministic results"""
    
    @pytest.fixture
    def test_config(self):
        """Load test configuration for system test"""
        return {
            "coordinator": {
                "mode": "backtest",
                "containers": ["data", "strategy", "risk", "execution"]
            },
            "data": {
                "source": "synthetic",
                "generator": "trending_market",
                "params": {
                    "start_price": 100.0,
                    "trend": 0.0002,  # 2% per 100 periods
                    "volatility": 0.01,
                    "periods": 252,  # 1 year of daily data
                    "seed": 42  # Deterministic generation
                }
            },
            "strategy": {
                "type": "momentum",
                "params": {
                    "lookback": 20,
                    "entry_threshold": 0.02,
                    "exit_threshold": -0.01
                }
            },
            "risk": {
                "initial_capital": 10000,
                "position_sizing": {
                    "method": "fixed_fractional",
                    "fraction": 0.1  # 10% per trade
                },
                "limits": {
                    "max_positions": 1,
                    "max_drawdown": 0.2  # 20% max drawdown
                }
            },
            "execution": {
                "mode": "backtest",
                "slippage": 0.0001,  # 1 basis point
                "commission": 0.001  # $0.001 per share
            }
        }
    
    @pytest.fixture
    def expected_results(self):
        """Load pre-computed expected results for this configuration"""
        return {
            "metrics": {
                "total_return": 0.0823,  # 8.23%
                "sharpe_ratio": 1.245,
                "max_drawdown": -0.0534,  # -5.34%
                "win_rate": 0.625,
                "total_trades": 16
            },
            "trades": [
                {
                    "timestamp": "2024-01-22T00:00:00",
                    "action": "BUY",
                    "symbol": "TEST",
                    "quantity": 98,
                    "price": 101.832,
                    "commission": 0.098
                },
                {
                    "timestamp": "2024-02-15T00:00:00",
                    "action": "SELL",
                    "symbol": "TEST",
                    "quantity": 98,
                    "price": 103.451,
                    "commission": 0.098
                }
                # ... more trades
            ],
            "final_portfolio": {
                "cash": 10823.45,
                "positions": {},
                "total_value": 10823.45
            }
        }
    
    def test_backtest_produces_exact_expected_results(self, test_config, expected_results):
        """Test that backtest produces exactly the expected results"""
        # Create coordinator
        coordinator = YAMLCoordinator()
        
        # Run backtest
        actual_results = coordinator.run_workflow(test_config)
        
        # Validate metrics match exactly
        actual_metrics = actual_results['metrics']
        expected_metrics = expected_results['metrics']
        
        assert abs(actual_metrics['total_return'] - expected_metrics['total_return']) < 1e-6
        assert abs(actual_metrics['sharpe_ratio'] - expected_metrics['sharpe_ratio']) < 1e-3
        assert abs(actual_metrics['max_drawdown'] - expected_metrics['max_drawdown']) < 1e-6
        assert actual_metrics['win_rate'] == expected_metrics['win_rate']
        assert actual_metrics['total_trades'] == expected_metrics['total_trades']
        
        # Validate trades match exactly
        actual_trades = actual_results['trades']
        expected_trades = expected_results['trades']
        
        assert len(actual_trades) == len(expected_trades)
        
        for actual, expected in zip(actual_trades, expected_trades):
            assert actual['timestamp'] == expected['timestamp']
            assert actual['action'] == expected['action']
            assert actual['symbol'] == expected['symbol']
            assert actual['quantity'] == expected['quantity']
            assert abs(actual['price'] - expected['price']) < 1e-6
            assert abs(actual['commission'] - expected['commission']) < 1e-6
        
        # Validate final portfolio state
        actual_portfolio = actual_results['final_portfolio']
        expected_portfolio = expected_results['final_portfolio']
        
        assert abs(actual_portfolio['cash'] - expected_portfolio['cash']) < 1e-2
        assert actual_portfolio['positions'] == expected_portfolio['positions']
        assert abs(actual_portfolio['total_value'] - expected_portfolio['total_value']) < 1e-2
    
    def test_backtest_reproducibility_across_runs(self, test_config):
        """Test that multiple runs produce identical results"""
        coordinator = YAMLCoordinator()
        
        # Run backtest 3 times
        results = []
        for i in range(3):
            result = coordinator.run_workflow(test_config)
            results.append(result)
        
        # All runs should produce identical results
        for i in range(1, 3):
            assert results[i]['metrics'] == results[0]['metrics']
            assert len(results[i]['trades']) == len(results[0]['trades'])
            
            # Check each trade matches
            for j, (trade1, trade2) in enumerate(zip(results[0]['trades'], results[i]['trades'])):
                assert trade1 == trade2, f"Trade {j} differs in run {i}"
    
    def test_container_isolation_maintained_throughout(self, test_config):
        """Test that container isolation is maintained during entire workflow"""
        # Create coordinator with isolation tracking
        coordinator = YAMLCoordinator(strict_isolation=True)
        
        # Add isolation monitor
        isolation_violations = []
        def track_violation(event):
            isolation_violations.append(event)
        
        coordinator.on_isolation_violation(track_violation)
        
        # Run backtest
        results = coordinator.run_workflow(test_config)
        
        # Verify no isolation violations
        assert len(isolation_violations) == 0, f"Found {len(isolation_violations)} isolation violations"
        
        # Verify results are still correct (isolation didn't affect functionality)
        assert results['metrics']['total_trades'] > 0
        assert results['final_portfolio']['total_value'] > 0
```

### Testing File Structure Template

For each step implementation, create this test structure:

```
tests/
├── fixtures/
│   ├── synthetic_data.py          # Synthetic data generators
│   ├── expected_results.py        # Pre-computed expected results
│   └── test_configs.py            # Test configuration templates
├── unit/
│   └── step_{N}/
│       ├── test_{component}_unit.py
│       └── conftest.py            # Step-specific fixtures
├── integration/
│   └── step_{N}/
│       ├── test_{flow}_integration.py
│       └── test_isolation.py      # Container isolation tests
└── system/
    └── step_{N}/
        ├── test_workflow_complete.py
        ├── test_reproducibility.py
        └── test_performance.py
```

### Test Execution Script

```python
# scripts/run_step_tests.py
import sys
import pytest
import argparse
from pathlib import Path

def run_step_tests(step_number: int, test_tier: str = "all"):
    """Run tests for a specific step and tier"""
    
    test_paths = {
        "unit": f"tests/unit/step_{step_number}/",
        "integration": f"tests/integration/step_{step_number}/",
        "system": f"tests/system/step_{step_number}/"
    }
    
    if test_tier == "all":
        paths_to_test = list(test_paths.values())
    else:
        paths_to_test = [test_paths[test_tier]]
    
    # Run tests with appropriate flags
    pytest_args = [
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "--strict-markers",  # Strict marker usage
        "--capture=no",  # Show print statements
    ]
    
    if test_tier in ["integration", "all"]:
        pytest_args.append("--validate-isolation")
    
    if test_tier in ["system", "all"]:
        pytest_args.extend(["--benchmark-only", "--benchmark-json=benchmark.json"])
    
    # Add coverage for unit tests
    if test_tier in ["unit", "all"]:
        pytest_args.extend([f"--cov=src", "--cov-report=term-missing"])
    
    # Run tests
    exit_code = pytest.main(pytest_args + paths_to_test)
    
    return exit_code == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for a specific step")
    parser.add_argument("step", type=int, help="Step number to test")
    parser.add_argument("--tier", choices=["unit", "integration", "system", "all"], 
                      default="all", help="Test tier to run")
    
    args = parser.parse_args()
    
    success = run_step_tests(args.step, args.tier)
    sys.exit(0 if success else 1)
```

---