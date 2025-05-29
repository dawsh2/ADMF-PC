"""
Simple test to verify walk-forward validation is working.
"""

import sys
sys.path.append('/Users/daws/ADMF-PC/ADMF-PC')

from src.strategy.optimization.walk_forward import (
    WalkForwardPeriod,
    WalkForwardValidator,
    WalkForwardAnalyzer
)

def test_walk_forward_period_generation():
    """Test that walk-forward periods are generated correctly."""
    print("Testing Walk-Forward Period Generation")
    print("=" * 50)
    
    # Test rolling walk-forward
    print("\n1. Rolling Walk-Forward (Fixed Window)")
    validator = WalkForwardValidator(
        data_length=1000,
        train_size=500,
        test_size=100,
        step_size=100,
        anchored=False
    )
    
    periods = validator.get_periods()
    print(f"Generated {len(periods)} periods:")
    
    for period in periods:
        print(f"  {period.period_id}: train[{period.train_start}:{period.train_end}], "
              f"test[{period.test_start}:{period.test_end}]")
    
    # Verify first period
    first = periods[0]
    assert first.train_start == 0
    assert first.train_end == 500
    assert first.test_start == 500
    assert first.test_end == 600
    print("✓ First period verified")
    
    # Verify rolling
    second = periods[1]
    assert second.train_start == 100  # Rolled forward by step_size
    assert second.train_end == 600
    print("✓ Rolling window verified")
    
    # Test anchored walk-forward
    print("\n2. Anchored Walk-Forward (Expanding Window)")
    anchored_validator = WalkForwardValidator(
        data_length=1000,
        train_size=500,
        test_size=100,
        step_size=100,
        anchored=True
    )
    
    anchored_periods = anchored_validator.get_periods()
    print(f"Generated {len(anchored_periods)} periods:")
    
    for period in anchored_periods[:3]:  # Show first 3
        print(f"  {period.period_id}: train[{period.train_start}:{period.train_end}] "
              f"(size={period.train_size}), test[{period.test_start}:{period.test_end}]")
    
    # Verify anchored behavior
    for period in anchored_periods:
        assert period.train_start == 0, "Anchored should always start from 0"
    print("✓ Anchored behavior verified")
    
    # Verify expanding window
    assert anchored_periods[0].train_size == 500
    assert anchored_periods[1].train_size == 600
    assert anchored_periods[2].train_size == 700
    print("✓ Expanding window verified")
    
    print("\n✅ All tests passed!")


def test_walk_forward_analyzer():
    """Test walk-forward analyzer functionality."""
    print("\n\nTesting Walk-Forward Analyzer")
    print("=" * 50)
    
    # Create simple validator
    validator = WalkForwardValidator(
        data_length=300,
        train_size=200,
        test_size=50,
        step_size=50,
        anchored=False
    )
    
    # Mock optimizer
    class MockOptimizer:
        def __init__(self):
            self.best_params = {'param1': 10}
            self.best_score = 1.5
            
        def optimize(self, func, space):
            # Simulate optimization
            return self.best_params
            
        def get_best_parameters(self):
            return self.best_params
            
        def get_best_score(self):
            return self.best_score
    
    # Mock objective
    class MockObjective:
        def calculate(self, results):
            return results.get('sharpe_ratio', 1.0)
    
    # Mock backtest function
    def mock_backtest(strategy_class, params, data):
        return {
            'sharpe_ratio': 1.2,
            'returns': [0.001, 0.002, -0.001],
            'total_return': 0.1,
            'max_drawdown': 0.05
        }
    
    # Create analyzer
    analyzer = WalkForwardAnalyzer(
        validator=validator,
        optimizer=MockOptimizer(),
        objective=MockObjective(),
        backtest_func=mock_backtest
    )
    
    # Test data slicing
    test_data = list(range(100))
    sliced = analyzer._slice_data(test_data, 10, 20)
    assert sliced == list(range(10, 20))
    print("✓ Data slicing works")
    
    # Test result aggregation
    mock_results = [
        {'train_performance': 1.5, 'test_performance': {'objective_score': 1.2}},
        {'train_performance': 1.6, 'test_performance': {'objective_score': 1.3}},
        {'train_performance': 1.4, 'test_performance': {'objective_score': 1.1}}
    ]
    
    aggregated = analyzer._aggregate_results(mock_results)
    print(f"\nAggregated results:")
    print(f"  Train mean: {aggregated['train']['mean']:.3f}")
    print(f"  Test mean: {aggregated['test']['mean']:.3f}")
    print(f"  Overfitting ratio: {aggregated['overfitting_ratio']:.3f}")
    
    assert aggregated['overfitting_ratio'] > 1.0, "Train should be better than test"
    print("✓ Result aggregation works")
    
    # Test summary creation
    summary = analyzer._create_summary(aggregated)
    print(f"\nSummary:")
    print(f"  Robust: {summary['robust']}")
    print(f"  Consistency: {summary['consistency']:.1%}")
    
    print("\n✅ Analyzer tests passed!")


def demonstrate_walk_forward_usage():
    """Demonstrate practical usage of walk-forward validation."""
    print("\n\nPractical Walk-Forward Example")
    print("=" * 50)
    
    # Scenario: Testing a strategy over 2 years of data
    # with 1 year training, 3 months test, rolling quarterly
    
    days_per_year = 252  # Trading days
    data_length = 2 * days_per_year  # 2 years
    train_days = days_per_year  # 1 year training
    test_days = 63  # ~3 months test
    step_days = 63  # Roll quarterly
    
    validator = WalkForwardValidator(
        data_length=data_length,
        train_size=train_days,
        test_size=test_days,
        step_size=step_days,
        anchored=False
    )
    
    periods = validator.get_periods()
    print(f"\nScenario: 2 years of data, 1 year train, 3 month test, quarterly roll")
    print(f"Generated {len(periods)} walk-forward periods:")
    
    for i, period in enumerate(periods):
        train_months = period.train_size / 21  # ~21 trading days per month
        test_months = period.test_size / 21
        print(f"\nQuarter {i+1}:")
        print(f"  Train: days {period.train_start}-{period.train_end} "
              f"({train_months:.1f} months)")
        print(f"  Test:  days {period.test_start}-{period.test_end} "
              f"({test_months:.1f} months)")
    
    print("\nThis ensures:")
    print("- Each quarter is tested on truly out-of-sample data")
    print("- Strategy parameters are re-optimized quarterly")
    print("- Performance is measured on future unseen data")
    print("- Overfitting is detected by comparing train vs test performance")


if __name__ == "__main__":
    test_walk_forward_period_generation()
    test_walk_forward_analyzer()
    demonstrate_walk_forward_usage()
    
    print("\n" + "=" * 50)
    print("Walk-Forward Validation is working correctly! ✅")
    print("=" * 50)