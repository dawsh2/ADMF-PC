"""
Container support for optimization.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

from ...core.containers import UniversalScopedContainer

logger = logging.getLogger(__name__)


class OptimizationContainer(UniversalScopedContainer):
    """Specialized container for optimization runs"""
    
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
    
    def create_trial_instance(self, parameters: Dict[str, Any]) -> tuple[str, Any]:
        """
        Create a new instance for parameter trial.
        
        Args:
            parameters: Trial parameters
            
        Returns:
            Tuple of (trial_id, component)
        """
        trial_id = f"{self.container_id}_trial_{self.trial_count}"
        self.trial_count += 1
        
        # Create component with trial parameters
        component_spec = self.base_config.copy()
        
        # Update parameters
        if 'params' in component_spec:
            component_spec['params'].update(parameters)
        else:
            component_spec['params'] = parameters
        
        # Add trial metadata
        component_spec['trial_id'] = trial_id
        component_spec['trial_number'] = self.trial_count
        
        # Fix the spec format for ComponentSpec
        if 'class' in component_spec and 'class_name' not in component_spec:
            component_spec['class_name'] = component_spec.pop('class')
        
        # Extract non-ComponentSpec fields and move to metadata
        extra_fields = {}
        component_spec_fields = {'name', 'class_name', 'params', 'capabilities', 'dependencies', 'config', 'metadata'}
        for key in list(component_spec.keys()):
            if key not in component_spec_fields:
                extra_fields[key] = component_spec.pop(key)
        
        # Store extra fields in metadata
        if extra_fields:
            if 'metadata' not in component_spec:
                component_spec['metadata'] = {}
            component_spec['metadata'].update(extra_fields)
        
        # For MockStrategy, create directly to avoid import issues
        if component_spec.get('class_name') == 'MockStrategy':
            # Import dynamically to avoid circular imports
            import sys
            if 'test_optimization_workflow' in sys.modules:
                MockStrategy = sys.modules['test_optimization_workflow'].MockStrategy
                component = MockStrategy(**parameters)
                
                # Add optimization capability if specified
                if 'capabilities' in component_spec and 'optimization' in component_spec['capabilities']:
                    from .capabilities import OptimizationCapability
                    capability = OptimizationCapability()
                    component = capability.apply(component, component_spec)
            else:
                # Fallback to basic mock
                class MockComponent:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                        self._parameters = kwargs
                    
                    def get_parameters(self):
                        return self._parameters
                
                component = MockComponent(**parameters)
        else:
            # Create isolated component instance using parent method
            # Use a unique name for each trial
            component_spec['name'] = f"component_{trial_id}"
            component = self.create_component(component_spec, initialize=True)
        
        logger.debug(f"Created trial instance {trial_id} with params: {parameters}")
        
        return trial_id, component
    
    def run_trial(self, parameters: Dict[str, Any], 
                  backtest_runner: Callable) -> Dict[str, Any]:
        """
        Run a single optimization trial.
        
        Args:
            parameters: Trial parameters
            backtest_runner: Function to run backtest with component
            
        Returns:
            Trial results
        """
        start_time = datetime.now()
        trial_id, component = self.create_trial_instance(parameters)
        
        try:
            # Initialize component if it has lifecycle
            if hasattr(component, 'initialize'):
                self.initialize_component(component)
            
            # Run backtest
            results = backtest_runner(component)
            
            # Add trial metadata
            results['trial_id'] = trial_id
            results['parameters'] = parameters.copy()
            results['duration'] = (datetime.now() - start_time).total_seconds()
            
            # Collect results
            self.results_collector.add_result(trial_id, parameters, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            
            # Return error result
            error_result = {
                'trial_id': trial_id,
                'parameters': parameters.copy(),
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            }
            
            self.results_collector.add_result(trial_id, parameters, error_result)
            
            raise
            
        finally:
            # Clean up trial instance
            if hasattr(component, 'teardown'):
                try:
                    component.teardown()
                except Exception as e:
                    logger.warning(f"Error during trial cleanup: {e}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get all optimization results"""
        return self.results_collector.get_all_results()
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get best result so far"""
        return self.results_collector.get_best_result()
    
    def clear_results(self) -> None:
        """Clear all results"""
        self.results_collector.clear()
        self.trial_count = 0


class OptimizationResultsCollector:
    """Collects and manages optimization results"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
    
    def add_result(self, trial_id: str, parameters: Dict[str, Any], 
                   results: Dict[str, Any]) -> None:
        """
        Add a trial result.
        
        Args:
            trial_id: Unique trial ID
            parameters: Trial parameters
            results: Trial results
        """
        # Create result record
        record = {
            'trial_id': trial_id,
            'parameters': parameters.copy(),
            'results': results,
            'timestamp': datetime.now()
        }
        
        self.results.append(record)
        
        # Update best if this is better
        if 'score' in results and results['score'] > self.best_score:
            self.best_score = results['score']
            self.best_result = record
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all collected results"""
        return self.results.copy()
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get best result"""
        return self.best_result.copy() if self.best_result else None
    
    def get_results_by_parameter(self, param_name: str, 
                                 param_value: Any) -> List[Dict[str, Any]]:
        """Get results filtered by parameter value"""
        filtered = []
        
        for result in self.results:
            params = result['parameters']
            if param_name in params and params[param_name] == param_value:
                filtered.append(result)
        
        return filtered
    
    def get_parameter_impact(self, param_name: str) -> Dict[str, Any]:
        """
        Analyze impact of a parameter on results.
        
        Args:
            param_name: Parameter to analyze
            
        Returns:
            Analysis of parameter impact
        """
        # Group results by parameter value
        value_groups = {}
        
        for result in self.results:
            if param_name in result['parameters']:
                value = result['parameters'][param_name]
                if value not in value_groups:
                    value_groups[value] = []
                
                if 'score' in result['results']:
                    value_groups[value].append(result['results']['score'])
        
        # Calculate statistics per value
        impact_analysis = {}
        
        for value, scores in value_groups.items():
            if scores:
                impact_analysis[value] = {
                    'count': len(scores),
                    'mean_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores)
                }
        
        return impact_analysis
    
    def clear(self) -> None:
        """Clear all results"""
        self.results.clear()
        self.best_result = None
        self.best_score = float('-inf')
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with results (if pandas available)
        """
        try:
            import pandas as pd
            
            # Flatten results for DataFrame
            rows = []
            
            for result in self.results:
                row = {
                    'trial_id': result['trial_id'],
                    'timestamp': result['timestamp']
                }
                
                # Add parameters
                for param_name, param_value in result['parameters'].items():
                    row[f'param_{param_name}'] = param_value
                
                # Add key results
                if 'results' in result:
                    results = result['results']
                    for key in ['score', 'sharpe_ratio', 'total_return', 
                               'max_drawdown', 'num_trades']:
                        if key in results:
                            row[key] = results[key]
                
                rows.append(row)
            
            return pd.DataFrame(rows)
            
        except ImportError:
            logger.warning("pandas not available for results analysis")
            return None