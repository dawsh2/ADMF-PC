"""
Regime-Adaptive Trading Workflow

Production-ready workflow that:
1. Detects historical regimes
2. Optimizes parameters per regime
3. Sets up live regime detection
4. Executes with adaptive parameters
"""

from typing import Dict, Any


def regime_adaptive_trading_workflow() -> Dict[str, Any]:
    """
    Define regime-adaptive trading workflow.
    
    This workflow creates a trading system that dynamically
    adjusts parameters based on detected market regimes.
    """
    return {
        'name': 'regime_adaptive_trading',
        'description': 'Detect regimes → Optimize per regime → Adaptive execution',
        'phases': [
            {
                'name': 'historical_regime_detection',
                'topology': 'analysis',
                'description': 'Detect regimes in historical data',
                'config_override': {
                    'analysis_type': 'regime_detection',
                    'classifiers': [
                        {'type': 'hmm_regime', 'n_states': 3},
                        {'type': 'volatility_regime', 'thresholds': {'low': 0.5, 'high': 1.5}}
                    ],
                    'lookback_periods': 500,
                    'output': './results/historical_regimes.json'
                }
            },
            {
                'name': 'regime_parameter_optimization',
                'topology': 'signal_generation',
                'description': 'Optimize parameters for each detected regime',
                'depends_on': ['historical_regime_detection'],
                'config_override': {
                    'mode': 'parameter_sweep',
                    'group_by': 'regime',
                    'regime_labels': '{phase.historical_regime_detection.output}',
                    'save_signals': True,
                    'signal_output_dir': './results/signals/by_regime/'
                }
            },
            {
                'name': 'ensemble_weight_optimization',
                'topology': 'signal_replay',
                'description': 'Optimize ensemble weights per regime',
                'depends_on': ['regime_parameter_optimization'],
                'config_override': {
                    'signal_directory': './results/signals/by_regime/',
                    'optimize': 'ensemble_weights',
                    'per_regime': True,
                    'regime_labels': '{phase.historical_regime_detection.output}',
                    'output': './results/ensemble_weights_by_regime.json'
                }
            },
            {
                'name': 'live_trading_simulation',
                'topology': 'backtest',
                'description': 'Simulate live trading with regime adaptation',
                'depends_on': ['ensemble_weight_optimization'],
                'config_override': {
                    'mode': 'paper_trading',
                    'regime_adaptive': True,
                    'regime_detector': {
                        'update_frequency': 'daily',
                        'classifiers': '{phase.historical_regime_detection.classifiers}'
                    },
                    'parameter_lookup': '{phase.ensemble_weight_optimization.output}',
                    'start_date': '{config.paper_trading_start}',
                    'end_date': '{config.paper_trading_end}'
                }
            }
        ],
        'outputs': {
            'regime_history': './results/historical_regimes.json',
            'optimal_parameters_by_regime': './results/optimal_params_by_regime.json',
            'ensemble_weights': './results/ensemble_weights_by_regime.json',
            'trading_performance': './results/regime_adaptive_performance.json'
        },
        'metadata': {
            'estimated_duration': '3-4 hours',
            'complexity': 'high',
            'recommended_for': ['changing market conditions', 'adaptive strategies', 'production']
        }
    }
