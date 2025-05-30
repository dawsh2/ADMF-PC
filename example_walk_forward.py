"""
Example of walk-forward validation in ADMF-PC.

This example demonstrates:
1. Setting up walk-forward validation
2. Running optimization on each training period
3. Testing on out-of-sample data
4. Analyzing results for robustness
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from src.strategy.optimization.walk_forward import (
    WalkForwardValidator,
    WalkForwardAnalyzer,
    ContainerizedWalkForward
)

from src.strategy.optimization import (
    GridOptimizer,
    BayesianOptimizer,
    SharpeObjective,
    CalmarObjective,
    CompositeObjective
)

from src.strategy.strategies.momentum import MomentumStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy


def generate_market_data(n_days: int = 1000) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    
    # Generate returns with different regimes
    returns = []
    
    # Trending regime (days 0-400)
    trend_returns = np.random.normal(0.001, 0.015, 400)
    
    # Volatile regime (days 400-700)
    volatile_returns = np.random.normal(0, 0.025, 300)
    
    # Mean-reverting regime (days 700-1000)
    mean_rev_returns = np.random.normal(0, 0.01, 300)
    
    all_returns = np.concatenate([trend_returns, volatile_returns, mean_rev_returns])
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(all_returns))
    
    # Create DataFrame
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * np.random.uniform(0.99, 1.01, n_days),
        'high': prices * np.random.uniform(1.0, 1.02, n_days),
        'low': prices * np.random.uniform(0.98, 1.0, n_days),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_days),
        'returns': all_returns
    })
    
    return df


def backtest_strategy(strategy_class: str, 
                     params: Dict[str, Any],
                     data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run backtest for a strategy on given data.
    
    This is a simplified backtest for demonstration.
    In production, this would use the full backtest engine.
    """
    # Create strategy instance
    if strategy_class == 'MomentumStrategy':
        strategy = MomentumStrategy(**params)
    elif strategy_class == 'MeanReversionStrategy':
        strategy = MeanReversionStrategy(**params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_class}")
    
    # Run simple backtest
    positions = []
    returns = []
    
    for i in range(len(data)):
        # Create market data dict
        market_data = {
            'symbol': 'TEST',
            'close': data.iloc[i]['close'],
            'high': data.iloc[i]['high'],
            'low': data.iloc[i]['low'],
            'volume': data.iloc[i]['volume'],
            'timestamp': data.iloc[i]['date']
        }
        
        # Generate signal
        signal = strategy.generate_signal(market_data)
        
        # Track positions and returns
        if signal:
            if signal['direction'] == 'BUY':
                positions.append(1)
            else:
                positions.append(-1)
        else:
            positions.append(0 if not positions else positions[-1])
        
        if i > 0 and positions[i-1] != 0:
            ret = positions[i-1] * data.iloc[i]['returns']
            returns.append(ret)
    
    # Calculate metrics
    if not returns:
        return {
            'returns': [0],
            'sharpe_ratio': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'num_trades': 0
        }
    
    returns = np.array(returns)
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + returns) - 1
    
    # Calculate drawdown
    rolling_max = np.maximum.accumulate(cum_returns + 1)
    drawdown = (cum_returns + 1) / rolling_max - 1
    max_drawdown = np.min(drawdown)
    
    # Calculate Sharpe ratio (annualized)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Count trades
    num_trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
    
    return {
        'returns': returns.tolist(),
        'sharpe_ratio': sharpe,
        'total_return': cum_returns[-1] if len(cum_returns) > 0 else 0,
        'max_drawdown': abs(max_drawdown),
        'num_trades': num_trades,
        'final_position': positions[-1] if positions else 0
    }


def run_walk_forward_example():
    """Run complete walk-forward validation example."""
    
    print("=== Walk-Forward Validation Example ===\n")
    
    # Generate market data
    print("1. Generating market data...")
    market_data = generate_market_data(1000)
    print(f"   Generated {len(market_data)} days of data")
    
    # Create walk-forward validator
    print("\n2. Setting up walk-forward validation...")
    validator = WalkForwardValidator(
        data_length=len(market_data),
        train_size=600,    # 600 days for training
        test_size=100,     # 100 days for testing
        step_size=100,     # Roll forward by 100 days
        anchored=False     # Use rolling window
    )
    
    periods = validator.get_periods()
    print(f"   Created {len(periods)} walk-forward periods:")
    for period in periods:
        print(f"   - {period.period_id}: train[{period.train_start}:{period.train_end}], "
              f"test[{period.test_start}:{period.test_end}]")
    
    # Create optimizer and objective
    print("\n3. Setting up optimization...")
    optimizer = GridOptimizer()
    
    # Use composite objective: 70% Sharpe, 30% Calmar
    objective = CompositeObjective([
        (SharpeObjective(), 0.7),
        (CalmarObjective(), 0.3)
    ])
    
    # Create analyzer
    analyzer = WalkForwardAnalyzer(
        validator=validator,
        optimizer=optimizer,
        objective=objective,
        backtest_func=lambda sc, p, d: backtest_strategy(sc, p, d)
    )
    
    # Define parameter space for momentum strategy
    parameter_space = {
        'lookback_period': [10, 20, 30, 40, 50],
        'momentum_threshold': [0.01, 0.015, 0.02, 0.025, 0.03],
        'rsi_period': [10, 14, 20]
    }
    
    base_params = {
        'signal_cooldown': 3600,  # 1 hour
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    # Run walk-forward analysis
    print("\n4. Running walk-forward analysis...")
    print("   This will optimize on each training period and test on out-of-sample data...")
    
    results = analyzer.analyze_strategy(
        strategy_class='MomentumStrategy',
        base_params=base_params,
        parameter_space=parameter_space,
        market_data=market_data
    )
    
    # Display results
    print("\n5. Walk-Forward Results:")
    print("=" * 60)
    
    for i, period_result in enumerate(results['periods']):
        print(f"\nPeriod {i}:")
        print(f"  Optimal Parameters:")
        for param, value in period_result['optimal_params'].items():
            if param not in base_params or base_params[param] != value:
                print(f"    - {param}: {value}")
        print(f"  Training Performance: {period_result['train_performance']:.3f}")
        print(f"  Test Performance: {period_result['test_performance']['objective_score']:.3f}")
        test_metrics = period_result['test_performance']['metrics']
        print(f"  Test Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
        print(f"  Test Total Return: {test_metrics['total_return']:.1%}")
        print(f"  Test Max Drawdown: {test_metrics['max_drawdown']:.1%}")
    
    # Display aggregate results
    print("\n6. Aggregate Results:")
    print("=" * 60)
    agg = results['aggregated']
    print(f"  Average Training Score: {agg['train']['mean']:.3f} ± {agg['train']['std']:.3f}")
    print(f"  Average Test Score: {agg['test']['mean']:.3f} ± {agg['test']['std']:.3f}")
    print(f"  Overfitting Ratio: {agg['overfitting_ratio']:.2f}")
    
    # Display summary
    print("\n7. Summary:")
    print("=" * 60)
    summary = results['summary']
    print(f"  Number of Periods: {summary['num_periods']}")
    print(f"  Average Train Score: {summary['avg_train_score']:.3f}")
    print(f"  Average Test Score: {summary['avg_test_score']:.3f}")
    print(f"  Consistency: {summary['consistency']:.1%}")
    print(f"  Strategy Robust: {'YES' if summary['robust'] else 'NO'}")
    
    # Save results
    from pathlib import Path
    results_path = Path("walk_forward_results.json")
    analyzer.save_results(results_path)
    print(f"\n8. Results saved to {results_path}")
    
    # Plot results
    plot_walk_forward_results(results, market_data)
    
    return results


def plot_walk_forward_results(results: Dict[str, Any], market_data: pd.DataFrame):
    """Plot walk-forward validation results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Market data with period markers
        ax1 = axes[0]
        ax1.plot(market_data.index, market_data['close'], 'b-', linewidth=0.5)
        ax1.set_title('Market Data with Walk-Forward Periods')
        ax1.set_ylabel('Price')
        
        # Mark train/test periods
        for i, period_result in enumerate(results['periods']):
            period = period_result['period']
            # Training period in light blue
            ax1.axvspan(period['train_start'], period['train_end'], 
                       alpha=0.2, color='blue', label='Train' if i == 0 else '')
            # Test period in light red
            ax1.axvspan(period['test_start'], period['test_end'], 
                       alpha=0.2, color='red', label='Test' if i == 0 else '')
        
        ax1.legend()
        
        # Plot 2: Performance by period
        ax2 = axes[1]
        periods = list(range(len(results['periods'])))
        train_scores = [p['train_performance'] for p in results['periods']]
        test_scores = [p['test_performance']['objective_score'] for p in results['periods']]
        
        x = np.array(periods)
        width = 0.35
        
        ax2.bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
        ax2.bar(x + width/2, test_scores, width, label='Test', alpha=0.7)
        
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Objective Score')
        ax2.set_title('Performance by Walk-Forward Period')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Optimal parameters over time
        ax3 = axes[2]
        
        # Extract parameter evolution
        lookback_periods = [p['optimal_params']['lookback_period'] 
                           for p in results['periods']]
        momentum_thresholds = [p['optimal_params']['momentum_threshold'] 
                              for p in results['periods']]
        
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(periods, lookback_periods, 'b-o', label='Lookback Period')
        line2 = ax3_twin.plot(periods, momentum_thresholds, 'r-s', label='Momentum Threshold')
        
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Lookback Period', color='b')
        ax3_twin.set_ylabel('Momentum Threshold', color='r')
        ax3.set_title('Optimal Parameters Evolution')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best')
        
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('walk_forward_analysis.png', dpi=300, bbox_inches='tight')
        print("\n9. Plots saved to walk_forward_analysis.png")
        
    except ImportError:
        print("\n9. Matplotlib not available, skipping plots")


def demonstrate_anchored_walk_forward():
    """Demonstrate anchored (expanding window) walk-forward."""
    
    print("\n=== Anchored Walk-Forward Example ===\n")
    
    # Generate data
    market_data = generate_market_data(1000)
    
    # Create anchored validator
    validator = WalkForwardValidator(
        data_length=len(market_data),
        train_size=400,    # Start with 400 days
        test_size=100,     # Test on 100 days
        step_size=100,     # Step by 100 days
        anchored=True      # Anchored (expanding window)
    )
    
    periods = validator.get_periods()
    print(f"Created {len(periods)} anchored periods:")
    for period in periods:
        print(f"  - {period.period_id}: train[{period.train_start}:{period.train_end}] "
              f"(size={period.train_size}), test[{period.test_start}:{period.test_end}]")
    
    print("\nNotice how training always starts from day 0 and expands!")


def demonstrate_containerized_walk_forward():
    """Demonstrate containerized walk-forward for production use."""
    
    print("\n=== Containerized Walk-Forward Example ===\n")
    
    # This would use the full container system in production
    print("In production, each walk-forward period would run in an isolated container:")
    print("  - Container 'walkforward_period_0_train' for training")
    print("  - Container 'walkforward_period_0_test' for testing")
    print("  - Complete isolation between periods")
    print("  - No state leakage")
    print("  - Parallel execution possible")
    
    # Mock container factory
    def mock_container_factory(container_id, config):
        print(f"  Created container: {container_id}")
        # In real implementation, this would create a UniversalScopedContainer
        return MagicMock()
    
    # Show how it would work
    from unittest.mock import MagicMock
    
    validator = WalkForwardValidator(
        data_length=1000,
        train_size=600,
        test_size=100,
        step_size=200,
        anchored=False
    )
    
    analyzer = WalkForwardAnalyzer(
        validator=validator,
        optimizer=GridOptimizer(),
        objective=SharpeObjective(),
        backtest_func=backtest_strategy
    )
    
    containerized = ContainerizedWalkForward(
        analyzer=analyzer,
        container_factory=mock_container_factory
    )
    
    print("\nContainers that would be created:")
    for period in validator.get_periods():
        print(f"\nPeriod {period.period_id}:")
        mock_container_factory(f"walkforward_{period.period_id}_train", {})
        mock_container_factory(f"walkforward_{period.period_id}_test", {})


if __name__ == "__main__":
    # Run main example
    results = run_walk_forward_example()
    
    # Show anchored example
    demonstrate_anchored_walk_forward()
    
    # Show containerized example
    demonstrate_containerized_walk_forward()
    
    print("\n=== Walk-Forward Validation Complete ===")
    
    # Key insights
    print("\nKey Insights:")
    print("1. Walk-forward validation tests strategy robustness over time")
    print("2. Each period optimizes on past data and tests on future data")
    print("3. Overfitting ratio > 1.5 suggests the strategy may not be robust")
    print("4. Consistent performance across periods indicates robustness")
    print("5. Container isolation ensures clean, reproducible results")