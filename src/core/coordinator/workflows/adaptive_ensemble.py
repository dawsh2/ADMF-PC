"""
Adaptive Ensemble Workflow

Complex multi-phase workflow that:
1. Grid search across parameter combinations
2. Analyze performance by regime
3. Optimize ensemble weights per regime
4. Validate with regime-adaptive strategy
"""

from typing import Dict, Any


def adaptive_ensemble_workflow() -> Dict[str, Any]:
    """
    Define the adaptive ensemble workflow.
    
    This workflow demonstrates how to combine multiple phases
    to create a sophisticated regime-adaptive trading system.
    """
    return {
        'name': 'adaptive_ensemble',
        'description': 'Grid search → Regime analysis → Ensemble optimization → Validation',
        'phases': [
            {
                'name': 'grid_search',
                'topology': 'signal_generation',
                'description': 'Run all parameter combinations using walk-forward validation',
                'config_override': {
                    'mode': 'walk_forward',
                    'save_signals': True,
                    'signal_output_dir': './results/signals/grid_search/',
                    'group_by_regime': True  # Group results by regime within each window
                }
            },
            {
                'name': 'regime_analysis',
                'topology': 'analysis',
                'description': 'Find best parameters per detected regime',
                'depends_on': ['grid_search'],
                'config_override': {
                    'analysis_type': 'regime_performance',
                    'input_data': '{phase.grid_search.output}',  # Use Phase 1 results
                    'output': './results/regime_configs.json'
                }
            },
            {
                'name': 'ensemble_optimization',
                'topology': 'signal_replay',
                'description': 'Find optimal ensemble weights per regime using walk-forward',
                'depends_on': ['regime_analysis'],
                'config_override': {
                    'mode': 'walk_forward',
                    'signal_directory': './results/signals/grid_search/',
                    'regime_configs': '{phase.regime_analysis.output}',
                    'optimize_weights': True,
                    'per_regime': True
                }
            },
            {
                'name': 'final_validation',
                'topology': 'backtest',
                'description': 'Deploy regime-adaptive ensemble on out-of-sample data',
                'depends_on': ['ensemble_optimization'],
                'config_override': {
                    'use_ensemble': True,
                    'regime_adaptive': True,
                    'ensemble_weights': '{phase.ensemble_optimization.output}',
                    'regime_configs': '{phase.regime_analysis.output}',
                    # Override dates for out-of-sample testing
                    'start_date': '{config.validation_start_date}',
                    'end_date': '{config.validation_end_date}'
                }
            }
        ],
        'outputs': {
            'signals': './results/signals/grid_search/',
            'regime_configs': './results/regime_configs.json',
            'ensemble_weights': './results/ensemble_weights.json',
            'final_performance': './results/adaptive_ensemble_performance.json'
        },
        'metadata': {
            'estimated_duration': '4-6 hours',
            'complexity': 'high',
            'recommended_for': ['regime changes', 'multi-strategy', 'walk-forward validation']
        }
    }
