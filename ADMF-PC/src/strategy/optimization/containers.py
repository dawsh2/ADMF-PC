"""
Container support for optimization workflows.

These containers provide isolated environments for running optimization
trials with proper state management and regime tracking.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import logging
import asyncio

from ...core.containers import UniversalScopedContainer
from ...core.components import ComponentFactory
from .protocols import OptimizationContainer as OptimizationContainerProtocol


logger = logging.getLogger(__name__)


class OptimizationContainer(UniversalScopedContainer):
    """
    Specialized container for optimization runs.
    
    Provides isolated execution environments for parameter trials
    with result collection and regime tracking.
    """
    
    def __init__(self, container_id: str, base_config: Dict[str, Any]):
        """
        Initialize optimization container.
        
        Args:
            container_id: Unique container ID
            base_config: Base configuration for components
        """
        super().__init__(container_id, container_type="optimization")
        self.base_config = base_config
        self.trial_count = 0
        self.results_collector = OptimizationResultsCollector()
        self.regime_tracker = RegimeTracker()
        self.active_trials = {}
    
    def create_trial_instance(self, 
                            parameters: Dict[str, Any],
                            trial_id: Optional[str] = None) -> Tuple[str, Any]:
        """
        Create isolated instance for parameter trial.
        
        Args:
            parameters: Trial parameters
            trial_id: Optional trial ID (generated if not provided)
            
        Returns:
            Tuple of (trial_id, component)
        """
        if trial_id is None:
            trial_id = f"{self.container_id}_trial_{self.trial_count}"
        
        self.trial_count += 1
        
        # Create component configuration
        component_spec = self.base_config.copy()
        
        # Update parameters
        if 'params' in component_spec:
            component_spec['params'].update(parameters)
        else:
            component_spec['params'] = parameters
        
        # Add trial metadata
        component_spec['metadata'] = component_spec.get('metadata', {})
        component_spec['metadata'].update({
            'trial_id': trial_id,
            'trial_number': self.trial_count,
            'parameters': parameters.copy()
        })
        
        # Create component with factory
        component = self.create_component(component_spec)
        
        # Track active trial
        self.active_trials[trial_id] = {
            'component': component,
            'parameters': parameters.copy(),
            'start_time': datetime.now(),
            'status': 'created'
        }
        
        logger.debug(f"Created trial instance {trial_id} with params: {parameters}")
        
        return trial_id, component
    
    def run_trial(self,
                 parameters: Dict[str, Any],
                 evaluator: Callable,
                 trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run optimization trial in isolation.
        
        Args:
            parameters: Trial parameters
            evaluator: Function to evaluate the trial
            trial_id: Optional trial ID
            
        Returns:
            Trial results with metrics
        """
        start_time = datetime.now()
        trial_id, component = self.create_trial_instance(parameters, trial_id)
        
        # Update trial status
        self.active_trials[trial_id]['status'] = 'running'
        
        try:
            # Initialize component if needed
            if hasattr(component, 'initialize'):
                self.initialize_component(component)
            
            # Run evaluation
            results = evaluator(component)
            
            # Ensure results is a dict
            if isinstance(results, (int, float)):
                results = {'score': results}
            
            # Add trial metadata
            results['trial_id'] = trial_id
            results['parameters'] = parameters.copy()
            results['duration'] = (datetime.now() - start_time).total_seconds()
            results['status'] = 'completed'
            
            # Add regime analysis if available
            if self.regime_tracker.has_data():
                results['regime_analysis'] = self.regime_tracker.analyze_trial(trial_id)
            
            # Collect results
            self.results_collector.add_result(trial_id, parameters, results)
            
            # Update trial status
            self.active_trials[trial_id]['status'] = 'completed'
            self.active_trials[trial_id]['results'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            
            # Create error result
            error_result = {
                'trial_id': trial_id,
                'parameters': parameters.copy(),
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds(),
                'status': 'failed'
            }
            
            # Update trial status
            self.active_trials[trial_id]['status'] = 'failed'
            self.active_trials[trial_id]['error'] = str(e)
            
            # Still collect the error result
            self.results_collector.add_result(trial_id, parameters, error_result)
            
            raise
            
        finally:
            # Cleanup trial
            self.cleanup_trial(trial_id)
    
    async def run_trial_async(self,
                            parameters: Dict[str, Any],
                            evaluator: Callable,
                            trial_id: Optional[str] = None) -> Dict[str, Any]:
        """Async version of run_trial for parallel execution."""
        return await asyncio.to_thread(self.run_trial, parameters, evaluator, trial_id)
    
    def get_trial_results(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get results for specific trial."""
        return self.results_collector.get_trial_result(trial_id)
    
    def cleanup_trial(self, trial_id: str) -> None:
        """Clean up resources for completed trial."""
        if trial_id in self.active_trials:
            trial_info = self.active_trials[trial_id]
            component = trial_info['component']
            
            # Teardown component if needed
            if hasattr(component, 'teardown'):
                try:
                    component.teardown()
                except Exception as e:
                    logger.warning(f"Error during trial {trial_id} cleanup: {e}")
            
            # Remove from active trials
            del self.active_trials[trial_id]
    
    def record_regime_change(self, regime: str, metadata: Dict[str, Any]) -> None:
        """Record regime change for analysis."""
        self.regime_tracker.record_regime_change(regime, metadata)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        return {
            'total_trials': self.trial_count,
            'completed_trials': len([t for t in self.active_trials.values() if t['status'] == 'completed']),
            'failed_trials': len([t for t in self.active_trials.values() if t['status'] == 'failed']),
            'best_result': self.results_collector.get_best_result(),
            'all_results': self.results_collector.get_all_results(),
            'regime_statistics': self.regime_tracker.get_statistics()
        }


class OptimizationResultsCollector:
    """Collects and manages optimization results."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.results_by_trial: Dict[str, Dict[str, Any]] = {}
        self.best_result: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
    
    def add_result(self, trial_id: str, parameters: Dict[str, Any], 
                   results: Dict[str, Any]) -> None:
        """Add a trial result."""
        record = {
            'trial_id': trial_id,
            'parameters': parameters.copy(),
            'results': results,
            'timestamp': datetime.now()
        }
        
        self.results.append(record)
        self.results_by_trial[trial_id] = record
        
        # Update best if this is better
        score = results.get('score')
        if score is not None:
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_result = record
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all collected results."""
        return self.results.copy()
    
    def get_trial_result(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get result for specific trial."""
        return self.results_by_trial.get(trial_id)
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get best result."""
        return self.best_result.copy() if self.best_result else None
    
    def get_results_by_regime(self, regime: str) -> List[Dict[str, Any]]:
        """Get results filtered by regime."""
        filtered = []
        
        for result in self.results:
            if 'regime_analysis' in result['results']:
                regime_data = result['results']['regime_analysis']
                if regime in regime_data.get('regimes', []):
                    filtered.append(result)
        
        return filtered
    
    def get_parameter_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter impact on results."""
        if not self.results:
            return {}
        
        # Group by each parameter
        param_impact = {}
        
        for result in self.results:
            params = result['parameters']
            score = result['results'].get('score')
            
            if score is not None:
                for param_name, param_value in params.items():
                    if param_name not in param_impact:
                        param_impact[param_name] = {}
                    
                    if param_value not in param_impact[param_name]:
                        param_impact[param_name][param_value] = []
                    
                    param_impact[param_name][param_value].append(score)
        
        # Calculate statistics
        analysis = {}
        for param_name, value_scores in param_impact.items():
            analysis[param_name] = {}
            
            for value, scores in value_scores.items():
                if scores:
                    analysis[param_name][value] = {
                        'count': len(scores),
                        'mean': sum(scores) / len(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'std': self._calculate_std(scores)
                    }
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


class RegimeTracker:
    """Tracks regime changes during optimization."""
    
    def __init__(self):
        self.regime_history: List[Dict[str, Any]] = []
        self.current_regime: Optional[str] = None
        self.regime_durations: Dict[str, List[float]] = {}
        self.regime_transitions: Dict[str, Dict[str, int]] = {}
    
    def record_regime_change(self, regime: str, metadata: Dict[str, Any]) -> None:
        """Record a regime change."""
        timestamp = datetime.now()
        
        # Record duration of previous regime
        if self.current_regime and self.regime_history:
            duration = (timestamp - self.regime_history[-1]['timestamp']).total_seconds()
            
            if self.current_regime not in self.regime_durations:
                self.regime_durations[self.current_regime] = []
            self.regime_durations[self.current_regime].append(duration)
            
            # Track transition
            if self.current_regime not in self.regime_transitions:
                self.regime_transitions[self.current_regime] = {}
            
            if regime not in self.regime_transitions[self.current_regime]:
                self.regime_transitions[self.current_regime][regime] = 0
            self.regime_transitions[self.current_regime][regime] += 1
        
        # Record new regime
        self.regime_history.append({
            'regime': regime,
            'timestamp': timestamp,
            'metadata': metadata.copy()
        })
        
        self.current_regime = regime
    
    def has_data(self) -> bool:
        """Check if regime data is available."""
        return len(self.regime_history) > 0
    
    def analyze_trial(self, trial_id: str) -> Dict[str, Any]:
        """Analyze regime context for a trial."""
        # This would correlate trial timing with regime history
        # For now, return current regime info
        return {
            'current_regime': self.current_regime,
            'regime_count': len(set(r['regime'] for r in self.regime_history)),
            'regimes': list(set(r['regime'] for r in self.regime_history))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get regime statistics."""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for record in self.regime_history:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        stats = {
            'regime_counts': regime_counts,
            'total_changes': len(self.regime_history) - 1,
            'current_regime': self.current_regime,
            'regime_durations': {
                regime: {
                    'mean': sum(durations) / len(durations) if durations else 0,
                    'total': sum(durations),
                    'count': len(durations)
                }
                for regime, durations in self.regime_durations.items()
            },
            'transitions': self.regime_transitions
        }
        
        return stats


class RegimeAwareOptimizationContainer(OptimizationContainer):
    """
    Extended optimization container with regime-specific tracking.
    
    This container provides enhanced support for the multi-pass
    optimization workflow described in the requirements.
    """
    
    def __init__(self, container_id: str, base_config: Dict[str, Any],
                 regime_classifiers: Optional[List[Any]] = None):
        """
        Initialize regime-aware optimization container.
        
        Args:
            container_id: Unique container ID
            base_config: Base configuration
            regime_classifiers: List of regime classifier components
        """
        super().__init__(container_id, base_config)
        self.regime_classifiers = regime_classifiers or []
        self.regime_performance: Dict[str, Dict[str, List[Dict]]] = {}
        self.parameter_by_regime: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    def run_trial_with_regime_tracking(self,
                                     parameters: Dict[str, Any],
                                     evaluator: Callable,
                                     trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run trial with detailed regime performance tracking.
        
        This method tracks performance metrics per regime during
        the trial, enabling regime-specific parameter analysis.
        """
        # Create wrapped evaluator that tracks regime performance
        def regime_aware_evaluator(component):
            # Run the actual evaluation
            results = evaluator(component)
            
            # Extract regime-specific performance if available
            if 'trades' in results and self.regime_tracker.has_data():
                regime_perf = self._analyze_regime_performance(results['trades'])
                results['regime_performance'] = regime_perf
                
                # Store for later analysis
                for regime, metrics in regime_perf.items():
                    if regime not in self.regime_performance:
                        self.regime_performance[regime] = {}
                    
                    param_key = str(parameters)
                    if param_key not in self.regime_performance[regime]:
                        self.regime_performance[regime][param_key] = []
                    
                    self.regime_performance[regime][param_key].append({
                        'trial_id': trial_id,
                        'metrics': metrics,
                        'parameters': parameters.copy()
                    })
            
            return results
        
        return self.run_trial(parameters, regime_aware_evaluator, trial_id)
    
    def _analyze_regime_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by regime."""
        regime_trades = {}
        
        # Group trades by regime
        for trade in trades:
            regime = trade.get('regime', 'unknown')
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)
        
        # Calculate metrics per regime
        regime_metrics = {}
        for regime, trades_in_regime in regime_trades.items():
            if trades_in_regime:
                returns = [t.get('return', 0) for t in trades_in_regime]
                regime_metrics[regime] = {
                    'trade_count': len(trades_in_regime),
                    'total_return': sum(returns),
                    'avg_return': sum(returns) / len(returns),
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'max_return': max(returns) if returns else 0,
                    'min_return': min(returns) if returns else 0
                }
        
        return regime_metrics
    
    def get_best_parameters_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """Get optimal parameters for each regime."""
        best_by_regime = {}
        
        for regime, param_results in self.regime_performance.items():
            best_score = None
            best_params = None
            
            for param_key, results in param_results.items():
                # Calculate average score for these parameters in this regime
                scores = [r['metrics'].get('total_return', 0) for r in results]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                if best_score is None or avg_score > best_score:
                    best_score = avg_score
                    best_params = results[0]['parameters'] if results else None
            
            if best_params:
                best_by_regime[regime] = {
                    'parameters': best_params,
                    'score': best_score,
                    'sample_size': len(param_results)
                }
        
        self.parameter_by_regime = best_by_regime
        return best_by_regime