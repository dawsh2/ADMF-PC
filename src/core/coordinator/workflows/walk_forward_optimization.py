"""
Walk-Forward Optimization Workflow

Standard walk-forward analysis workflow that:
1. Optimizes parameters on training window
2. Tests on out-of-sample window
3. Steps forward and repeats
"""

from typing import Dict, Any


def walk_forward_optimization_workflow() -> Dict[str, Any]:
    """
    Define walk-forward optimization workflow.
    
    This is a single-phase workflow that uses walk-forward
    sequence execution within the backtest topology.
    """
    return {
        'name': 'walk_forward_optimization',
        'description': 'Rolling window parameter optimization with out-of-sample testing',
        'phases': [
            {
                'name': 'walk_forward_backtest',
                'topology': 'backtest',
                'sequence': 'walk_forward',  # Specifies sequence type
                'description': 'Walk-forward optimization and testing',
                'config_override': {
                    'mode': 'walk_forward',
                    'walk_forward': {
                        'train_periods': '{config.train_periods}',  # e.g., 252 days
                        'test_periods': '{config.test_periods}',    # e.g., 63 days
                        'step_size': '{config.step_size}',          # e.g., 21 days
                        'optimization_metric': '{config.optimization_metric}'  # e.g., 'sharpe_ratio'
                    },
                    'save_window_results': True,
                    'aggregate_results': True
                }
            }
        ],
        'outputs': {
            'window_results': './results/walk_forward/window_results/',
            'aggregated_performance': './results/walk_forward/aggregated_performance.json',
            'optimal_parameters': './results/walk_forward/optimal_parameters_by_window.json'
        },
        'metadata': {
            'estimated_duration': '2-3 hours',
            'complexity': 'medium',
            'recommended_for': ['parameter stability testing', 'realistic backtesting']
        }
    }
