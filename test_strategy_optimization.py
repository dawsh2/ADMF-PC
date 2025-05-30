"""
Test the strategy optimization framework.

This verifies that all the optimization components work together properly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, Optional
from datetime import datetime

# Test imports
print("Testing imports...")

try:
    # Protocols
    from src.strategy.protocols import Strategy, Indicator, Classifier
    from src.strategy.optimization.protocols import Optimizer, Objective, Constraint
    
    # Components
    from src.strategy.components.indicators import SimpleMovingAverage, RSI
    from src.strategy.components.classifiers import TrendClassifier
    from src.strategy.components.signal_replay import SignalCapture, SignalReplayer
    
    # Optimization
    from src.strategy.optimization.optimizers import GridOptimizer, RandomOptimizer
    from src.strategy.optimization.objectives import SharpeObjective, CompositeObjective
    from src.strategy.optimization.constraints import RangeConstraint, RelationalConstraint
    
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_basic_components():
    """Test basic strategy components."""
    print("\nTesting basic components...")
    
    # Test indicator
    sma = SimpleMovingAverage(period=3)
    for i, price in enumerate([100, 102, 101, 103, 99]):
        value = sma.calculate(price)
        if i >= 2:  # Should have value after 3 periods
            assert value is not None, f"SMA should have value after period {i+1}"
            print(f"  SMA({i+1}): {value:.2f}")
    
    # Test classifier
    classifier = TrendClassifier()
    for price in [100, 102, 104, 106, 108]:
        classification = classifier.classify({'close': price})
        print(f"  Classification: {classification} (confidence: {classifier.confidence:.2f})")
    
    print("✓ Basic components working")


def test_optimization_framework():
    """Test optimization framework."""
    print("\nTesting optimization framework...")
    
    # Create optimizer
    optimizer = GridOptimizer()
    
    # Create objective
    objective = SharpeObjective()
    
    # Define evaluation function
    def evaluate_params(params: Dict[str, Any]) -> float:
        # Mock evaluation
        period = params.get('period', 20)
        threshold = params.get('threshold', 0.01)
        
        # Simulate that period=30 and threshold=0.02 is optimal
        distance = abs(period - 30) + abs(threshold - 0.02) * 100
        score = 2.0 - distance * 0.1  # Max score of 2.0
        
        mock_results = {
            'sharpe_ratio': score,
            'returns': [0.001] * 100
        }
        
        return objective.calculate(mock_results)
    
    # Define parameter space
    param_space = {
        'period': [10, 20, 30, 40],
        'threshold': [0.01, 0.02, 0.03]
    }
    
    # Run optimization
    best_params = optimizer.optimize(
        evaluate_func=evaluate_params,
        parameter_space=param_space
    )
    
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {optimizer.get_best_score():.3f}")
    print(f"  Total trials: {len(optimizer.get_optimization_history())}")
    
    # Verify best parameters are close to optimal
    assert best_params['period'] == 30, "Should find optimal period"
    assert best_params['threshold'] == 0.02, "Should find optimal threshold"
    
    print("✓ Optimization framework working")


def test_constraints():
    """Test constraint system."""
    print("\nTesting constraints...")
    
    # Create constraints
    range_constraint = RangeConstraint('period', min_value=10, max_value=50)
    relational_constraint = RelationalConstraint('fast_period', '<', 'slow_period')
    
    # Test range constraint
    params1 = {'period': 60}
    assert not range_constraint.is_satisfied(params1), "Should violate max constraint"
    adjusted1 = range_constraint.validate_and_adjust(params1)
    assert adjusted1['period'] == 50, "Should clip to max value"
    print(f"  Range constraint: {params1} -> {adjusted1}")
    
    # Test relational constraint
    params2 = {'fast_period': 30, 'slow_period': 20}
    assert not relational_constraint.is_satisfied(params2), "Should violate relational constraint"
    adjusted2 = relational_constraint.validate_and_adjust(params2)
    assert adjusted2['fast_period'] < adjusted2['slow_period'], "Should fix relationship"
    print(f"  Relational constraint: {params2} -> {adjusted2}")
    
    print("✓ Constraint system working")


def test_signal_capture_replay():
    """Test signal capture and replay."""
    print("\nTesting signal capture and replay...")
    
    # Create signal capture
    capture = SignalCapture("test_capture")
    
    # Capture some signals
    from src.strategy.protocols import SignalDirection
    
    for i in range(5):
        signal = {
            'symbol': 'TEST',
            'direction': SignalDirection.BUY if i % 2 == 0 else SignalDirection.SELL,
            'strength': 0.5 + i * 0.1,
            'price': 100 + i,
            'timestamp': datetime.now()
        }
        capture.capture_signal(signal, source=f"strategy_{i % 2}")
    
    print(f"  Captured {len(capture.signals)} signals")
    
    # Create replayer
    replayer = SignalReplayer(capture)
    
    # Test replay with weights
    from src.strategy.components.signal_replay import WeightedSignalAggregator
    aggregator = WeightedSignalAggregator()
    
    weights = {'strategy_0': 0.7, 'strategy_1': 0.3}
    replayed_signals = replayer.replay_with_weights(weights, aggregator)
    
    print(f"  Replayed {len(replayed_signals)} aggregated signals")
    
    print("✓ Signal capture/replay working")


def test_composite_objective():
    """Test composite objectives."""
    print("\nTesting composite objectives...")
    
    from src.strategy.optimization.objectives import MaxReturnObjective, MinDrawdownObjective
    
    # Create composite objective
    composite = CompositeObjective([
        (SharpeObjective(), 0.5),
        (MaxReturnObjective(), 0.3),
        (MinDrawdownObjective(), 0.2)
    ])
    
    # Test with mock results
    results = {
        'sharpe_ratio': 1.5,
        'total_return': 0.20,
        'max_drawdown': -0.10,
        'returns': [0.001] * 100
    }
    
    score = composite.calculate(results)
    print(f"  Composite score: {score:.3f}")
    
    # Test requirements
    requirements = composite.get_requirements()
    print(f"  Required fields: {requirements}")
    
    print("✓ Composite objectives working")


def run_all_tests():
    """Run all tests."""
    print("ADMF-PC Strategy Optimization Framework Tests")
    print("=" * 50)
    
    try:
        test_basic_components()
        test_optimization_framework()
        test_constraints()
        test_signal_capture_replay()
        test_composite_objective()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        
    except AssertionError as e:
        print(f"\n✗ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)