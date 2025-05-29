"""
Example usage of the optimization framework.

This demonstrates how to use the protocol-based optimization system
with container isolation, regime analysis, and signal replay.
"""

from typing import Dict, Any, List, Optional
import logging

from ...core.coordinator import ApplicationCoordinator
from ..protocols import Strategy, Indicator, Classifier, SignalDirection
from ..components.indicators import SimpleMovingAverage, RSI
from ..components.classifiers import TrendClassifier, create_market_regime_classifier
from ..components.signal_replay import SignalCapture, SignalReplayer, WeightedSignalAggregator
from .optimizers import GridOptimizer, RandomOptimizer
from .objectives import SharpeObjective, CompositeObjective, MinDrawdownObjective
from .constraints import RelationalConstraint, RangeConstraint, CompositeConstraint
from .workflows import (
    ContainerizedComponentOptimizer,
    SequentialOptimizationWorkflow,
    RegimeBasedOptimizationWorkflow
)

logger = logging.getLogger(__name__)


def example_basic_optimization():
    """
    Example of basic strategy optimization.
    
    This shows how to optimize a simple moving average crossover strategy
    using grid search with container isolation.
    """
    print("\n=== Basic Strategy Optimization ===")
    
    # Define strategy component specification
    ma_crossover_spec = {
        'type': 'ma_crossover',
        'parameter_space': {
            'fast_period': [10, 20, 30, 40, 50],
            'slow_period': [50, 100, 150, 200],
            'threshold': [0.001, 0.002, 0.005, 0.01]
        },
        'constraints': [
            {'type': 'relational', 'param1': 'fast_period', 'relation': '<', 'param2': 'slow_period'},
            {'type': 'range', 'param': 'threshold', 'min': 0.0001, 'max': 0.1}
        ]
    }
    
    # Create optimizer and objective
    optimizer = GridOptimizer()
    objective = SharpeObjective(risk_free_rate=0.02)
    
    # Create containerized optimizer
    container_optimizer = ContainerizedComponentOptimizer(
        optimizer=optimizer,
        objective=objective,
        use_containers=True  # Enable container isolation
    )
    
    # Mock backtest runner
    def run_backtest(params: Dict[str, Any]) -> Dict[str, Any]:
        # This would be your actual backtesting logic
        import random
        import math
        
        # Simulate results based on parameters
        fast = params.get('fast_period', 20)
        slow = params.get('slow_period', 50)
        threshold = params.get('threshold', 0.001)
        
        # Better parameters should give better results
        param_quality = (slow - fast) / slow * (1 - threshold * 100)
        
        returns = [random.gauss(0.0005 * param_quality, 0.01) for _ in range(252)]
        
        # Calculate metrics
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)
        sharpe = (avg_return * 252) / (std_dev * math.sqrt(252)) if std_dev > 0 else 0
        
        return {
            'returns': returns,
            'sharpe_ratio': sharpe,
            'total_return': sum(returns),
            'max_drawdown': -0.1,
            'num_trades': int(100 * param_quality)
        }
    
    # Run optimization
    results = container_optimizer.optimize_component(
        component_spec=ma_crossover_spec,
        backtest_runner=run_backtest,
        n_trials=None  # Use all parameter combinations
    )
    
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best Sharpe ratio: {results['best_score']:.3f}")
    print(f"Total trials: {len(results['optimization_history'])}")
    print(f"Optimization time: {results['duration']:.1f} seconds")


def example_sequential_optimization():
    """
    Example of multi-stage sequential optimization.
    
    This demonstrates optimizing indicators first, then strategy parameters,
    with results feeding forward between stages.
    """
    print("\n=== Sequential Multi-Stage Optimization ===")
    
    # Define optimization stages
    stages = [
        {
            'name': 'indicator_optimization',
            'component': {
                'type': 'rsi_indicator',
                'parameter_space': {
                    'period': [7, 14, 21, 28],
                    'overbought': [65, 70, 75, 80],
                    'oversold': [20, 25, 30, 35]
                }
            },
            'optimizer': {'type': 'grid'},
            'objective': {'type': 'sharpe'},
            'n_trials': 100,
            'feed_forward': True
        },
        {
            'name': 'strategy_optimization',
            'component': {
                'type': 'rsi_strategy',
                'parameter_space': {
                    'stop_loss': [0.01, 0.02, 0.03, 0.05],
                    'take_profit': [0.02, 0.04, 0.06, 0.08],
                    'position_size': [0.1, 0.2, 0.3, 0.5]
                }
            },
            'optimizer': {'type': 'bayesian'},
            'objective': {
                'type': 'composite',
                'components': [
                    {'type': 'sharpe', 'weight': 0.5},
                    {'type': 'drawdown', 'weight': 0.3},
                    {'type': 'return', 'weight': 0.2}
                ]
            },
            'n_trials': 200,
            'use_previous_best': True,  # Use best indicator params
            'inherit_parameters': True
        }
    ]
    
    # Create workflow
    workflow = SequentialOptimizationWorkflow(stages)
    
    # Run workflow
    results = workflow.run()
    
    # Display results
    for stage_name, stage_results in results.items():
        if 'error' not in stage_results:
            print(f"\nStage: {stage_name}")
            print(f"  Best parameters: {stage_results.get('best_parameters', {})}")
            print(f"  Best score: {stage_results.get('best_score', 0):.3f}")


def example_regime_optimization():
    """
    Example of regime-based optimization.
    
    This shows how to optimize separately for different market regimes,
    creating an adaptive strategy that switches parameters based on
    market conditions.
    """
    print("\n=== Regime-Based Optimization ===")
    
    # Configure regime detector
    regime_detector_config = {
        'type': 'composite_classifier',
        'classifiers': [
            {'type': 'trend', 'weight': 0.6},
            {'type': 'volatility', 'weight': 0.4}
        ]
    }
    
    # Configure component to optimize
    component_config = {
        'type': 'adaptive_momentum',
        'parameter_space': {
            'lookback': [10, 20, 40, 60],
            'entry_threshold': [0.01, 0.02, 0.03],
            'exit_threshold': [0.005, 0.01, 0.015],
            'volatility_filter': [1.0, 1.5, 2.0]
        }
    }
    
    # Configure optimizer
    optimizer_config = {
        'type': 'bayesian',
        'objective': 'sharpe',
        'n_trials_per_regime': 100
    }
    
    # Create workflow
    workflow = RegimeBasedOptimizationWorkflow(
        regime_detector_config=regime_detector_config,
        component_config=component_config,
        optimizer_config=optimizer_config
    )
    
    # Run workflow
    results = workflow.run()
    
    # Display results
    print(f"\nDetected regimes: {results['detected_regimes']}")
    
    for regime in results['detected_regimes']:
        if regime in results:
            regime_results = results[regime]
            print(f"\nRegime: {regime}")
            print(f"  Best parameters: {regime_results.get('best_parameters', {})}")
            print(f"  Best Sharpe: {regime_results.get('best_score', 0):.3f}")
    
    print(f"\nAdaptive configuration created: {results['adaptive_config']}")


def example_signal_replay_optimization():
    """
    Example of signal replay optimization.
    
    This demonstrates capturing signals from multiple strategies
    and optimizing their weights without re-running backtests.
    """
    print("\n=== Signal Replay Weight Optimization ===")
    
    # Create some example strategies
    class MomentumStrategy:
        def __init__(self, period: int = 20):
            self.period = period
            self.indicator = SimpleMovingAverage(period)
            
        def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            price = market_data.get('close', 0)
            self.indicator.update(price)
            
            if not self.indicator.ready:
                return None
            
            # Simple momentum signal
            if price > self.indicator.value * 1.01:
                return {
                    'symbol': market_data['symbol'],
                    'direction': SignalDirection.BUY,
                    'strength': min((price / self.indicator.value - 1) * 100, 1.0),
                    'price': price,
                    'timestamp': market_data.get('timestamp')
                }
            elif price < self.indicator.value * 0.99:
                return {
                    'symbol': market_data['symbol'],
                    'direction': SignalDirection.SELL,
                    'strength': min((1 - price / self.indicator.value) * 100, 1.0),
                    'price': price,
                    'timestamp': market_data.get('timestamp')
                }
            
            return None
    
    class RSIStrategy:
        def __init__(self, period: int = 14):
            self.rsi = RSI(period)
            
        def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            price = market_data.get('close', 0)
            self.rsi.update(price)
            
            if not self.rsi.ready:
                return None
            
            rsi_value = self.rsi.value
            
            # RSI signals
            if rsi_value < 30:  # Oversold
                return {
                    'symbol': market_data['symbol'],
                    'direction': SignalDirection.BUY,
                    'strength': (30 - rsi_value) / 30,
                    'price': price,
                    'timestamp': market_data.get('timestamp')
                }
            elif rsi_value > 70:  # Overbought
                return {
                    'symbol': market_data['symbol'],
                    'direction': SignalDirection.SELL,
                    'strength': (rsi_value - 70) / 30,
                    'price': price,
                    'timestamp': market_data.get('timestamp')
                }
            
            return None
    
    # Create strategies
    strategies = [
        ('momentum_fast', MomentumStrategy(10)),
        ('momentum_slow', MomentumStrategy(30)),
        ('rsi', RSIStrategy(14))
    ]
    
    # Generate mock market data
    import random
    from datetime import datetime, timedelta
    
    market_data = []
    base_price = 100
    current_time = datetime.now()
    
    for i in range(500):
        # Random walk
        change = random.gauss(0, 0.01)
        base_price *= (1 + change)
        
        market_data.append({
            'symbol': 'TEST',
            'close': base_price,
            'timestamp': current_time + timedelta(hours=i)
        })
    
    # Phase 1: Capture signals
    print("\n1. Capturing signals from strategies...")
    capture = SignalCapture('test_capture')
    
    for data in market_data:
        for name, strategy in strategies:
            signal = strategy.generate_signal(data)
            if signal:
                capture.capture_signal(signal, name)
    
    print(f"   Total signals captured: {len(capture.signals)}")
    for name, _ in strategies:
        count = len(capture.get_signals_by_source(name))
        print(f"   {name}: {count} signals")
    
    # Phase 2: Optimize weights
    print("\n2. Optimizing signal weights...")
    
    # Create replayer and aggregator
    replayer = SignalReplayer(capture)
    aggregator = WeightedSignalAggregator(min_signals=1, min_agreement=0.0)
    
    # Define weight optimization
    weight_space = {
        'momentum_fast': (0.0, 1.0),
        'momentum_slow': (0.0, 1.0),
        'rsi': (0.0, 1.0)
    }
    
    # Mock performance calculator
    def calculate_performance(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simulate performance from signals
        if not signals:
            return {'sharpe_ratio': 0, 'total_return': 0}
        
        # Simple mock calculation
        num_signals = len(signals)
        avg_strength = sum(s['strength'] for s in signals) / num_signals
        
        return {
            'sharpe_ratio': avg_strength * 1.5,  # Mock
            'total_return': num_signals * avg_strength * 0.001,
            'num_signals': num_signals
        }
    
    # Optimize weights
    optimizer = RandomOptimizer(seed=42)
    
    best_weights = None
    best_score = -float('inf')
    
    for _ in range(100):  # 100 trials
        # Sample weights
        weights = {}
        for strategy_name, _ in strategies:
            weights[strategy_name] = random.uniform(0, 1)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Replay and evaluate
        signals = replayer.replay_with_weights(weights, aggregator)
        performance = calculate_performance(signals)
        score = performance['sharpe_ratio']
        
        if score > best_score:
            best_score = score
            best_weights = weights
    
    print(f"\n   Best weights found:")
    for name, weight in best_weights.items():
        print(f"   {name}: {weight:.3f}")
    print(f"   Best Sharpe ratio: {best_score:.3f}")


def example_with_constraints():
    """
    Example showing how to use constraints in optimization.
    
    This demonstrates parameter constraints, relational constraints,
    and performance constraints.
    """
    print("\n=== Optimization with Constraints ===")
    
    # Create constraints
    constraints = [
        # Parameter ranges
        RangeConstraint('fast_period', min_value=5, max_value=50),
        RangeConstraint('slow_period', min_value=20, max_value=200),
        
        # Relational constraint
        RelationalConstraint('fast_period', '<', 'slow_period'),
        
        # Custom functional constraint
        FunctionalConstraint(
            lambda p: p.get('slow_period', 0) - p.get('fast_period', 0) >= 10,
            description="slow_period must be at least 10 greater than fast_period"
        )
    ]
    
    # Create composite constraint
    composite_constraint = CompositeConstraint(constraints)
    
    # Test parameters
    test_params = {
        'fast_period': 20,
        'slow_period': 25,  # Violates constraint
        'threshold': 0.01
    }
    
    print("Testing parameters:", test_params)
    print("Constraint satisfied:", composite_constraint.is_satisfied(test_params))
    
    # Adjust parameters
    adjusted = composite_constraint.validate_and_adjust(test_params)
    print("Adjusted parameters:", adjusted)
    print("Constraint satisfied after adjustment:", composite_constraint.is_satisfied(adjusted))


def run_all_examples():
    """Run all optimization examples."""
    try:
        example_basic_optimization()
    except Exception as e:
        print(f"Basic optimization example failed: {e}")
    
    try:
        example_sequential_optimization()
    except Exception as e:
        print(f"Sequential optimization example failed: {e}")
    
    try:
        example_regime_optimization()
    except Exception as e:
        print(f"Regime optimization example failed: {e}")
    
    try:
        example_signal_replay_optimization()
    except Exception as e:
        print(f"Signal replay optimization example failed: {e}")
    
    try:
        example_with_constraints()
    except Exception as e:
        print(f"Constraints example failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("ADMF-PC Strategy Optimization Examples")
    print("=" * 50)
    
    run_all_examples()
    
    print("\n" + "=" * 50)
    print("Examples complete!")