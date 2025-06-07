"""
Adaptive Strategy Search workflow.

This is a composable workflow that keeps searching for profitable
strategies until it finds ones that meet the target criteria.

It demonstrates the power of composable workflows for automated
strategy research.
"""

from typing import Dict, Any, List, Optional
import random
from ..protocols import WorkflowProtocol, PhaseConfig, PhaseEnhancerProtocol, WorkflowBranch


class AdaptiveStrategySearchWorkflow:
    """
    Composable workflow that searches for profitable strategies.
    
    This workflow:
    1. Runs parameter optimization
    2. Checks if results meet target criteria
    3. If not, expands search space and tries again
    4. Can branch to different search strategies
    5. Stops when finding strategies that meet targets
    
    Perfect for automated strategy research!
    """
    
    def __init__(self, enhancers: Optional[List[PhaseEnhancerProtocol]] = None):
        """
        Initialize with optional phase enhancers.
        
        Args:
            enhancers: List of components that can enhance phase configs
        """
        self.enhancers = enhancers or []
        self.search_history = []  # Track what we've tried
        
        # Workflow defaults
        self.defaults = {
            'target_sharpe': 1.5,  # Keep searching until Sharpe > 1.5
            'target_win_rate': 0.55,  # And win rate > 55%
            'max_iterations': 10,  # Maximum search iterations
            'expansion_factor': 1.5,  # How much to expand search space
            'trace_level': 'minimal',
            'objective_function': {'name': 'sharpe_ratio'},
            'parameter_selection': {
                'method': 'top_n',
                'n': 10  # Keep top 10 for analysis
            }
        }
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        """
        Create phases for strategy search.
        
        Uses train/test optimization workflow as base.
        """
        phases = {
            "parameter_search": PhaseConfig(
                name="parameter_search",
                sequence="train_test",
                topology="backtest",
                description="Search parameter space for profitable strategies",
                config={
                    **config,
                    'optimization_phase': True,
                    'parameter_space': config.get('parameter_space', {}),
                    'objective_function': config.get('objective_function', self.defaults['objective_function'])
                },
                output={
                    'optimal_parameters': True,
                    'parameter_performance': True,
                    'all_results': True  # Keep all results for analysis
                }
            ),
            "strategy_analysis": PhaseConfig(
                name="strategy_analysis",
                sequence="single_pass",
                topology="analysis",
                description="Analyze found strategies",
                config={
                    'analysis_type': 'strategy_characteristics',
                    'metrics': ['sharpe_ratio', 'win_rate', 'max_drawdown', 'profit_factor']
                },
                input={
                    'strategies': '{parameter_search.output.optimal_parameters}',
                    'performance': '{parameter_search.output.parameter_performance}'
                },
                output={
                    'strategy_summary': True,
                    'meets_criteria': True
                },
                depends_on=['parameter_search']
            )
        }
        
        # Apply enhancers
        for enhancer in self.enhancers:
            phases = enhancer.enhance(phases)
        
        return phases
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """
        Continue searching if we haven't found good enough strategies.
        
        Args:
            result: Result from previous iteration
            iteration: Current iteration number
            
        Returns:
            True if should continue searching
        """
        # Check if we found strategies meeting criteria
        analysis = result.get('phase_results', {}).get('strategy_analysis', {})
        if analysis.get('output', {}).get('meets_criteria', False):
            return False  # Found good strategies!
        
        # Check iteration limit
        max_iterations = result.get('config', {}).get('max_iterations', self.defaults['max_iterations'])
        if iteration >= max_iterations - 1:
            return False  # Hit limit
        
        # Check if we have any promising leads
        best_sharpe = self._get_best_sharpe(result)
        if best_sharpe > 0 and iteration > 3:
            # Getting somewhere but slowly - maybe try different approach
            return True
        
        return True  # Keep searching
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """
        Expand parameter search space for next iteration.
        
        Args:
            config: Current configuration
            result: Result from previous iteration
            iteration: Current iteration number
            
        Returns:
            Modified configuration with expanded search space
        """
        new_config = config.copy()
        
        # Analyze what didn't work
        best_sharpe = self._get_best_sharpe(result)
        
        # Expand parameter space
        if 'parameter_space' in new_config:
            expansion_factor = config.get('expansion_factor', self.defaults['expansion_factor'])
            
            for strategy, params in new_config['parameter_space'].get('strategies', {}).items():
                for param, values in params.items():
                    if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                        # Expand range
                        min_val = min(values)
                        max_val = max(values)
                        range_size = max_val - min_val
                        
                        # Add values outside current range
                        new_values = values.copy()
                        
                        # Expand upward if best performance was at upper bound
                        if self._best_was_at_boundary(result, strategy, param, 'upper'):
                            new_values.extend([
                                max_val + range_size * 0.5,
                                max_val + range_size * 1.0
                            ])
                        
                        # Expand downward if best was at lower bound
                        if self._best_was_at_boundary(result, strategy, param, 'lower'):
                            new_values.extend([
                                max(0, min_val - range_size * 0.5),
                                max(0, min_val - range_size * 1.0)
                            ])
                        
                        # Add intermediate values if sparse
                        if len(values) < 5:
                            for i in range(len(values) - 1):
                                new_values.append((values[i] + values[i+1]) / 2)
                        
                        params[param] = sorted(list(set(new_values)))
        
        # Track what we've tried
        self.search_history.append({
            'iteration': iteration,
            'best_sharpe': best_sharpe,
            'parameter_space_size': self._count_combinations(new_config)
        })
        
        return new_config
    
    def get_branches(self, result: Dict[str, Any]) -> Optional[List[WorkflowBranch]]:
        """
        Branch to different search strategies if stuck.
        
        Args:
            result: Current workflow result
            
        Returns:
            List of possible branches
        """
        branches = []
        
        # If we're not finding anything good, try different approach
        best_sharpe = self._get_best_sharpe(result)
        
        if best_sharpe < 0.5 and len(self.search_history) > 2:
            # Current approach isn't working - try regime-based
            branches.append(WorkflowBranch(
                condition=lambda r: True,
                workflow='regime_adaptive_ensemble',
                config_modifier=lambda c, r: {
                    **c,
                    'regime_detection': {'n_regimes': 3},
                    'note': 'Switched to regime-based approach due to poor results'
                }
            ))
        
        elif 0.5 <= best_sharpe < 1.0:
            # Getting somewhere - try ensemble approach
            branches.append(WorkflowBranch(
                condition=lambda r: True,
                workflow='train_test_optimization',
                config_modifier=lambda c, r: {
                    **c,
                    'ensemble_weights': True,
                    'note': 'Adding ensemble optimization'
                }
            ))
        
        return branches if branches else None
    
    def _get_best_sharpe(self, result: Dict[str, Any]) -> float:
        """Extract best Sharpe ratio from results."""
        try:
            # Try to get from parameter search results
            param_results = result.get('phase_results', {}).get('parameter_search', {})
            performance = param_results.get('output', {}).get('parameter_performance', {})
            
            if performance:
                return max(p.get('sharpe_ratio', -999) for p in performance.values())
                
            # Fallback to aggregated results
            return result.get('aggregated_results', {}).get('best_metric', -999)
        except:
            return -999
    
    def _best_was_at_boundary(self, result: Dict[str, Any], strategy: str, 
                             param: str, boundary: str) -> bool:
        """Check if best result was at parameter boundary."""
        # This would need to analyze the parameter_performance data
        # For now, use simple heuristic
        return random.random() > 0.7  # 30% chance to expand in each direction
    
    def _count_combinations(self, config: Dict[str, Any]) -> int:
        """Count total parameter combinations."""
        total = 1
        for strategy, params in config.get('parameter_space', {}).get('strategies', {}).items():
            for param, values in params.items():
                if isinstance(values, list):
                    total *= len(values)
        return total