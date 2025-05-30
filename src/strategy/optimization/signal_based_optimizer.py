"""
Signal-based optimization using the three-pattern architecture.

This module shows how to integrate signal generation and replay
into the optimization workflow for fast parameter tuning.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import json
from pathlib import Path
import numpy as np

from ...execution import (
    SignalGenerationContainer,
    SignalReplayContainer,
    SignalGenerationContainerFactory,
    SignalReplayContainerFactory
)
from ..components import SignalCapture
from .protocols import OptimizationObjective, OptimizationResult


class SignalBasedOptimizer:
    """
    Optimizer that uses signal generation and replay for fast optimization.
    
    This implements a two-phase approach:
    1. Generate signals once for all strategy variants
    2. Replay signals with different parameters for fast optimization
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: str = "output/optimization"
    ):
        """
        Initialize signal-based optimizer.
        
        Args:
            base_config: Base configuration
            output_dir: Directory for output files
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Signal storage
        self._signal_logs: Dict[str, str] = {}  # strategy_id -> signal_log_path
        self._analysis_results: Dict[str, Dict[str, Any]] = {}
        
    async def generate_signals_for_strategies(
        self,
        strategy_configs: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
        symbols: List[str]
    ) -> Dict[str, str]:
        """
        Generate signals for multiple strategy configurations.
        
        Args:
            strategy_configs: List of strategy configurations
            start_date: Start date for signal generation
            end_date: End date for signal generation
            symbols: Symbols to analyze
            
        Returns:
            Mapping of strategy_id to signal log path
        """
        print(f"Generating signals for {len(strategy_configs)} strategies...")
        
        for i, strategy_config in enumerate(strategy_configs):
            strategy_id = strategy_config.get('id', f'strategy_{i}')
            print(f"\nGenerating signals for {strategy_id}...")
            
            # Create configuration for signal generation
            signal_gen_config = {
                'data_config': self.base_config.get('data', {}),
                'indicator_config': self.base_config.get('indicators', {}),
                'classifiers': self.base_config.get('classifiers', []),
                'strategies': [strategy_config],
                'analysis_config': {
                    'lookback_bars': 20,
                    'forward_bars': [1, 5, 10, 20],
                    'analysis_types': ['mae_mfe', 'forward_returns', 'signal_quality']
                }
            }
            
            # Create container
            container = SignalGenerationContainerFactory.create_instance(signal_gen_config)
            
            # Run analysis
            results = await container.run_analysis(start_date, end_date, symbols)
            
            # Save signals
            signal_log_path = self.output_dir / f"signals_{strategy_id}.json"
            container.export_signals(str(signal_log_path))
            
            # Store results
            self._signal_logs[strategy_id] = str(signal_log_path)
            self._analysis_results[strategy_id] = results
            
            print(f"✓ Generated {results['total_signals_analyzed']} signals for {strategy_id}")
            
        return self._signal_logs
        
    async def optimize_ensemble_weights(
        self,
        signal_logs: Dict[str, str],
        weight_combinations: Optional[List[Dict[str, float]]] = None,
        objective: Optional[OptimizationObjective] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Optimize ensemble weights using signal replay.
        
        Args:
            signal_logs: Mapping of strategy_id to signal log path
            weight_combinations: List of weight combinations to test
            objective: Optimization objective (default: Sharpe ratio)
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        if weight_combinations is None:
            # Generate default combinations
            weight_combinations = self._generate_weight_combinations(
                list(signal_logs.keys())
            )
            
        print(f"\nOptimizing ensemble weights over {len(weight_combinations)} combinations...")
        
        best_weights = {}
        best_score = -np.inf
        
        # Merge all signal logs for ensemble
        merged_signals = self._merge_signal_logs(signal_logs)
        merged_log_path = self.output_dir / "merged_signals.json"
        with open(merged_log_path, 'w') as f:
            json.dump(merged_signals, f)
            
        for i, weights in enumerate(weight_combinations):
            print(f"\rTesting combination {i+1}/{len(weight_combinations)}", end='')
            
            # Create replay configuration
            replay_config = {
                'signal_log_path': str(merged_log_path),
                'weight_config': {
                    'strategy_weights': weights,
                    'aggregation_method': 'weighted_vote'
                },
                'risk_config': self.base_config.get('risk', {}),
                'execution_config': self.base_config.get('execution', {})
            }
            
            # Create container and run replay
            container = SignalReplayContainerFactory.create_instance(replay_config)
            
            results = await container.run_replay(
                self.base_config['data']['start_date'],
                self.base_config['data']['end_date']
            )
            
            # Calculate score
            if objective:
                score = objective.calculate(results)
            else:
                # Default to Sharpe ratio
                score = results['performance_metrics']['sharpe_ratio']
                
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                
        print(f"\n\n✓ Best weights: {best_weights}")
        print(f"✓ Best score: {best_score:.4f}")
        
        return best_weights, best_score
        
    async def optimize_risk_parameters(
        self,
        signal_log: str,
        risk_param_grid: Dict[str, List[Any]],
        fixed_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize risk parameters using signal replay.
        
        Args:
            signal_log: Path to signal log
            risk_param_grid: Grid of risk parameters to test
            fixed_weights: Fixed ensemble weights (optional)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print("\nOptimizing risk parameters...")
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(risk_param_grid)
        
        best_params = {}
        best_score = -np.inf
        
        for params in param_combinations:
            # Create replay configuration with risk parameters
            replay_config = {
                'signal_log_path': signal_log,
                'weight_config': {
                    'strategy_weights': fixed_weights or {},
                },
                'risk_config': {
                    'risk_parameters': params
                },
                'execution_config': self.base_config.get('execution', {})
            }
            
            # Run replay
            container = SignalReplayContainerFactory.create_instance(replay_config)
            results = await container.run_replay(
                self.base_config['data']['start_date'],
                self.base_config['data']['end_date']
            )
            
            # Calculate score
            score = results['performance_metrics']['sharpe_ratio']
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        print(f"✓ Best risk parameters: {best_params}")
        print(f"✓ Best score: {best_score:.4f}")
        
        return best_params, best_score
        
    def get_signal_quality_report(self) -> Dict[str, Any]:
        """Get signal quality analysis for all strategies."""
        report = {}
        
        for strategy_id, results in self._analysis_results.items():
            if 'strategy_metrics' in results:
                metrics = results['strategy_metrics'].get(strategy_id, {})
                report[strategy_id] = {
                    'total_signals': metrics.get('total_signals', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'expectancy': metrics.get('expectancy', 0),
                    'avg_mae': metrics.get('avg_mae', 0),
                    'avg_mfe': metrics.get('avg_mfe', 0),
                    'mae_mfe_ratio': metrics.get('mae_mfe_ratio', 0),
                    'signal_strength': {
                        'mean': metrics.get('signal_strength_avg', 0),
                        'std': metrics.get('signal_strength_std', 0)
                    }
                }
                
        return report
        
    def _generate_weight_combinations(
        self,
        strategy_ids: List[str],
        granularity: float = 0.1
    ) -> List[Dict[str, float]]:
        """Generate weight combinations for strategies."""
        if len(strategy_ids) == 1:
            return [{strategy_ids[0]: 1.0}]
            
        combinations = []
        
        # Generate combinations with given granularity
        if len(strategy_ids) == 2:
            for w1 in np.arange(0, 1 + granularity, granularity):
                w2 = 1 - w1
                if w2 >= 0:
                    combinations.append({
                        strategy_ids[0]: round(w1, 2),
                        strategy_ids[1]: round(w2, 2)
                    })
        else:
            # For more strategies, use some predefined combinations
            # Full grid would be too large
            # Equal weights
            equal_weight = 1.0 / len(strategy_ids)
            combinations.append({
                sid: equal_weight for sid in strategy_ids
            })
            
            # Dominant strategies
            for dominant_id in strategy_ids:
                weights = {sid: 0.1 for sid in strategy_ids}
                weights[dominant_id] = 1 - 0.1 * (len(strategy_ids) - 1)
                combinations.append(weights)
                
        return combinations
        
    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations from grid."""
        import itertools
        
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        return combinations
        
    def _merge_signal_logs(
        self,
        signal_logs: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Merge multiple signal logs into one."""
        merged_signals = []
        
        for strategy_id, log_path in signal_logs.items():
            with open(log_path, 'r') as f:
                signals = json.load(f)
                
            # Add strategy ID to each signal
            for signal in signals:
                signal['strategy_id'] = strategy_id
                merged_signals.append(signal)
                
        # Sort by timestamp
        merged_signals.sort(key=lambda x: x['timestamp'])
        
        return merged_signals


async def example_signal_based_optimization():
    """Example of signal-based optimization workflow."""
    # Configuration
    base_config = {
        'data': {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 12, 31),
            'symbols': ['AAPL', 'GOOGL', 'MSFT']
        },
        'indicators': {
            'SMA_20': {'period': 20},
            'SMA_50': {'period': 50},
            'RSI_14': {'period': 14}
        },
        'risk': {
            'max_position_size': 0.02,
            'max_total_exposure': 0.1
        }
    }
    
    # Strategy configurations to test
    strategy_configs = [
        {
            'id': 'momentum_fast',
            'class': 'MomentumStrategy',
            'parameters': {'fast_period': 10, 'slow_period': 20}
        },
        {
            'id': 'momentum_slow',
            'class': 'MomentumStrategy',
            'parameters': {'fast_period': 20, 'slow_period': 50}
        },
        {
            'id': 'mean_reversion',
            'class': 'MeanReversionStrategy',
            'parameters': {'lookback': 20, 'z_threshold': 2.0}
        }
    ]
    
    # Create optimizer
    optimizer = SignalBasedOptimizer(base_config)
    
    # Phase 1: Generate signals
    signal_logs = await optimizer.generate_signals_for_strategies(
        strategy_configs,
        base_config['data']['start_date'],
        base_config['data']['end_date'],
        base_config['data']['symbols']
    )
    
    # Get signal quality report
    quality_report = optimizer.get_signal_quality_report()
    print("\nSignal Quality Report:")
    for strategy_id, metrics in quality_report.items():
        print(f"\n{strategy_id}:")
        print(f"  Win rate: {metrics['win_rate']:.2%}")
        print(f"  Expectancy: {metrics['expectancy']:.4f}")
        print(f"  MAE/MFE ratio: {metrics['mae_mfe_ratio']:.2f}")
    
    # Phase 2: Optimize ensemble weights
    best_weights, best_score = await optimizer.optimize_ensemble_weights(signal_logs)
    
    # Phase 3: Optimize risk parameters with best weights
    best_risk_params, final_score = await optimizer.optimize_risk_parameters(
        list(signal_logs.values())[0],  # Use first signal log
        risk_param_grid={
            'max_position_size': [0.01, 0.02, 0.03],
            'max_total_exposure': [0.05, 0.1, 0.15]
        },
        fixed_weights=best_weights
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best ensemble weights: {best_weights}")
    print(f"Best risk parameters: {best_risk_params}")
    print(f"Final Sharpe ratio: {final_score:.4f}")
    
    return best_weights, best_risk_params


if __name__ == "__main__":
    # Run example
    asyncio.run(example_signal_based_optimization())