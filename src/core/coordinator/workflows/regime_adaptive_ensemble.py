"""
Regime-Adaptive Ensemble workflow implementation.

This is a complex multi-phase workflow that:
1. Runs parameter optimization with walk-forward validation
2. Analyzes performance by regime
3. Optimizes ensemble weights per regime
4. Validates the adaptive ensemble

This demonstrates the full power of workflow composition.
"""

from typing import Dict, Any, List, Optional
from ..protocols import WorkflowProtocol, PhaseConfig, PhaseEnhancerProtocol


class RegimeAdaptiveEnsembleWorkflow:
    """
    Regime-Adaptive Ensemble workflow using Protocol + Composition.
    
    This high-level workflow finds optimal parameters per regime,
    then optimizes ensemble weights per regime, with final validation.
    
    Phases:
    1. Parameter Grid Search (walk-forward, signal generation)
    2. Regime Analysis (analysis only, no execution)  
    3. Ensemble Optimization (walk-forward, signal replay)
    4. Final Validation (full backtest)
    """
    
    def __init__(self, enhancers: Optional[List[PhaseEnhancerProtocol]] = None):
        """
        Initialize with optional phase enhancers.
        
        Args:
            enhancers: List of components that can enhance phase configs
        """
        self.enhancers = enhancers or []
        
        # Workflow defaults
        self.defaults = {
            'walk_forward': {
                'window_size': 252,  # 1 year windows
                'step_size': 63,     # 3 month steps
                'min_periods': 3     # At least 3 windows
            },
            'regime_detection': {
                'method': 'hmm',
                'n_regimes': 3
            },
            'ensemble_optimization': {
                'method': 'mean_variance',
                'constraints': {
                    'sum_to_one': True,
                    'long_only': True,
                    'min_weight': 0.0,
                    'max_weight': 0.5
                }
            },
            'validation_split': 0.2,  # 20% for final test
            'trace_level': 'minimal',  # Memory efficient
            'objective_function': {'name': 'sharpe_ratio'}
        }
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        """
        Convert user config to phase definitions.
        
        Creates four phases with complex inter-dependencies.
        
        Args:
            config: User configuration
            
        Returns:
            Dict with all phase configurations
        """
        # Calculate validation split
        train_end = self._calculate_train_end(config)
        
        phases = {
            # Phase 1: Parameter Grid Search with Walk-Forward
            "parameter_grid_search": PhaseConfig(
                name="parameter_grid_search",
                sequence="walk_forward",
                topology="signal_generation",  # Generate and store signals
                description="Walk-forward parameter optimization with signal capture",
                config={
                    **config,
                    'data': {
                        **config.get('data', {}),
                        'end': train_end  # Don't use test data
                    },
                    'walk_forward': config.get('walk_forward', self.defaults['walk_forward']),
                    'parameter_space': config.get('parameter_space', {}),
                    'objective_function': config.get('objective_function', self.defaults['objective_function']),
                    'signal_output_dir': './signals/grid_search/',
                    'capture_regime_labels': True
                },
                output={
                    'signal_store': './signals/grid_search/',
                    'performance_by_window': True,
                    'performance_by_regime': True,
                    'regime_labels': True,
                    'parameter_rankings': True
                }
            ),
            
            # Phase 2: Regime Analysis (no execution)
            "regime_analysis": PhaseConfig(
                name="regime_analysis",
                sequence="regime_analysis",  # Special analysis sequence
                topology="analysis",         # Analysis topology (no trading)
                description="Identify best parameters per regime",
                config={
                    'regime_detection': config.get('regime_detection', self.defaults['regime_detection']),
                    'min_samples_per_regime': 20,
                    'performance_threshold': 0.5  # Min Sharpe to consider
                },
                input={
                    'performance_data': '{parameter_grid_search.output.performance_by_window}',
                    'regime_labels': '{parameter_grid_search.output.regime_labels}',
                    'parameter_rankings': '{parameter_grid_search.output.parameter_rankings}'
                },
                output={
                    'regime_optimal_params': True,
                    'regime_transitions': True,
                    'regime_statistics': True
                },
                depends_on=['parameter_grid_search']
            ),
            
            # Phase 3: Ensemble Weight Optimization
            "ensemble_optimization": PhaseConfig(
                name="ensemble_optimization",
                sequence="walk_forward",
                topology="signal_replay",  # Replay signals, don't regenerate
                description="Find optimal ensemble weights per regime",
                config={
                    **config,
                    'data': {
                        **config.get('data', {}),
                        'end': train_end
                    },
                    'walk_forward': config.get('walk_forward', self.defaults['walk_forward']),
                    'signal_input_dir': './signals/grid_search/',
                    'ensemble_optimization': config.get('ensemble_optimization', 
                                                      self.defaults['ensemble_optimization']),
                    'optimize_per_regime': True
                },
                input={
                    'signal_store': '{parameter_grid_search.output.signal_store}',
                    'regime_params': '{regime_analysis.output.regime_optimal_params}',
                    'regime_transitions': '{regime_analysis.output.regime_transitions}'
                },
                output={
                    'regime_ensemble_weights': True,
                    'ensemble_performance': True,
                    'weight_stability': True
                },
                depends_on=['regime_analysis']
            ),
            
            # Phase 4: Final Out-of-Sample Validation
            "final_validation": PhaseConfig(
                name="final_validation",
                sequence="single_pass",
                topology="backtest",  # Full backtest with regime switching
                description="Test regime-adaptive ensemble on held-out data",
                config={
                    **config,
                    'data': {
                        **config.get('data', {}),
                        'start': train_end  # Use only test data
                    },
                    'execution': {
                        'regime_switching': True,
                        'rebalance_on_regime_change': True,
                        'regime_detection_lag': 5  # Days to detect regime
                    },
                    'results': {
                        'store_trades': True,
                        'store_equity_curve': True,  # Want full curve for final
                        'store_regime_history': True
                    }
                },
                input={
                    'regime_params': '{regime_analysis.output.regime_optimal_params}',
                    'regime_weights': '{ensemble_optimization.output.regime_ensemble_weights}',
                    'regime_model': '{regime_analysis.output.regime_model}'
                },
                output={
                    'final_metrics': True,
                    'regime_performance': True,
                    'trade_analysis': True,
                    'equity_curve': True
                },
                depends_on=['ensemble_optimization']
            )
        }
        
        # Apply enhancers
        for enhancer in self.enhancers:
            phases = enhancer.enhance(phases)
        
        return phases
    
    def _calculate_train_end(self, config: Dict[str, Any]) -> str:
        """Calculate end date for training (start of test period)."""
        from datetime import datetime, timedelta
        
        data_config = config.get('data', {})
        end_date = data_config.get('end', data_config.get('end_date'))
        
        if isinstance(end_date, str):
            end = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end = end_date
        
        # Calculate validation split
        validation_split = config.get('validation_split', self.defaults['validation_split'])
        
        # Simple approach - go back by percentage of year
        # More sophisticated would look at actual data range
        days_back = int(365 * validation_split)
        train_end = end - timedelta(days=days_back)
        
        return train_end.strftime('%Y-%m-%d')