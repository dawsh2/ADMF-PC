"""
Execution mode handlers for different workflow patterns.

This module contains the implementation for signal generation and
signal replay modes that were previously in main.py.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from ...execution import (
    SignalGenerationContainer,
    SignalReplayContainer,
    SignalGenerationContainerFactory,
    SignalReplayContainerFactory
)


class ExecutionModeHandler:
    """Handles different execution modes for the system."""
    
    @staticmethod
    async def run_signal_generation(
        base_config: Dict[str, Any],
        signal_output: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run signal generation mode for analysis.
        
        Args:
            base_config: Base configuration
            signal_output: Path to save generated signals
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary
        """
        print("Running in Signal Generation mode...")
        
        # Extract configuration
        data_config = base_config.get('data', {})
        strategies = base_config.get('strategies', [])
        indicators = base_config.get('indicators', [])
        classifiers = base_config.get('classifiers', [])
        
        # Create signal generation configuration
        signal_gen_config = {
            'data_config': data_config,
            'indicator_config': {'indicators': indicators},
            'classifiers': classifiers,
            'strategies': strategies,
            'analysis_config': {
                'lookback_bars': 20,
                'forward_bars': [1, 5, 10, 20],
                'analysis_types': ['mae_mfe', 'forward_returns', 'signal_quality', 'correlation']
            }
        }
        
        # Create signal generation container
        container = SignalGenerationContainerFactory.create_instance(signal_gen_config)
        
        # Run analysis
        start_date = data_config.get('start_date')
        end_date = data_config.get('end_date')
        symbols = data_config.get('symbols', [])
        
        results = await container.run_analysis(start_date, end_date, symbols)
        
        # Save signals if output path specified
        if signal_output:
            container.export_signals(signal_output)
            print(f"Signals saved to: {signal_output}")
        
        # Print analysis results
        ExecutionModeHandler._print_signal_analysis_results(results, container)
        
        return {
            'success': True,
            'mode': 'signal_generation',
            'results': results,
            'signal_output': signal_output
        }
    
    @staticmethod
    async def run_signal_replay(
        base_config: Dict[str, Any],
        signal_log: str,
        weights: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run signal replay mode for ensemble optimization.
        
        Args:
            base_config: Base configuration
            signal_log: Path to signal log file
            weights: Strategy weights (JSON string or file path)
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary
        """
        if not signal_log:
            raise ValueError("signal_log is required for signal-replay mode")
        
        print(f"Running in Signal Replay mode with signals from: {signal_log}")
        
        # Parse strategy weights
        strategy_weights = {}
        if weights:
            if weights.startswith('{'):
                # JSON string
                strategy_weights = json.loads(weights)
            elif Path(weights).exists():
                # JSON file
                with open(weights, 'r') as f:
                    strategy_weights = json.load(f)
            else:
                raise ValueError(f"Invalid weights specification: {weights}")
        else:
            # Default equal weights
            print("No weights specified, using equal weights")
            strategy_weights = {}  # Will be set to equal by the container
        
        # Extract risk configuration from base config
        risk_config = base_config.get('risk', {
            'max_position_size': 0.02,
            'max_total_exposure': 0.1
        })
        
        # Create signal replay configuration
        replay_config = {
            'signal_log_path': signal_log,
            'weight_config': {
                'strategy_weights': strategy_weights,
                'aggregation_method': 'weighted_vote',
                'min_agreement': 0.5
            },
            'risk_config': {
                'risk_parameters': risk_config
            },
            'execution_config': {
                'initial_balance': base_config.get('capital', 100000),
                'slippage_model': None,
                'commission_model': None
            }
        }
        
        # Create signal replay container
        container = SignalReplayContainerFactory.create_instance(replay_config)
        
        # Run replay
        data_config = base_config.get('data', {})
        start_date = data_config.get('start_date')
        end_date = data_config.get('end_date')
        
        results = await container.run_replay(start_date, end_date)
        
        # Print results
        ExecutionModeHandler._print_signal_replay_results(results)
        
        return {
            'success': True,
            'mode': 'signal_replay',
            'results': results,
            'signal_log': signal_log,
            'weights': strategy_weights
        }
    
    @staticmethod
    def _print_signal_analysis_results(results: Dict[str, Any], container: Any) -> None:
        """Print signal analysis results."""
        print("\nSignal Analysis Results:")
        print(f"Total signals analyzed: {results['total_signals_analyzed']}")
        print("\nStrategy Performance:")
        for strategy_id, metrics in results['strategy_metrics'].items():
            print(f"\n{strategy_id}:")
            print(f"  Total signals: {metrics['total_signals']}")
            print(f"  Win rate: {metrics['win_rate']:.2%}")
            print(f"  Expectancy: {metrics['expectancy']:.4f}")
            print(f"  Average MAE: {metrics.get('avg_mae', 0):.4f}")
            print(f"  Average MFE: {metrics.get('avg_mfe', 0):.4f}")
        
        # Get optimal stops
        optimal_stops = container.get_optimal_stops()
        print(f"\nOptimal Stop Loss: {optimal_stops.get('optimal_stop_loss', 0):.2%}")
        print(f"Optimal Take Profit: {optimal_stops.get('optimal_take_profit', 0):.2%}")
    
    @staticmethod
    def _print_signal_replay_results(results: Dict[str, Any]) -> None:
        """Print signal replay results."""
        print("\nSignal Replay Results:")
        print(f"Signals processed: {results['signals_processed']}")
        print(f"Orders generated: {results['orders_generated']}")
        print(f"Fills executed: {results['fills_executed']}")
        print(f"Final portfolio value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total return: {results['performance_metrics']['total_return']:.2%}")
        print(f"Win rate: {results['performance_metrics']['win_rate']:.2%}")
        print(f"Sharpe ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
        
        print("\nEnsemble Weights Used:")
        for strategy, weight in results['ensemble_weights'].items():
            print(f"  {strategy}: {weight:.2%}")