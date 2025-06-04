"""
Parameter analysis for multi-parameter workflow detection.

Analyzes configuration to detect when multi-parameter execution is needed.
"""

import logging
from typing import Dict, Any, List
import itertools

from ....types.workflow import WorkflowConfig

logger = logging.getLogger(__name__)


class ParameterAnalyzer:
    """Analyzes parameters to detect multi-parameter workflow requirements."""
    
    def requires_multi_parameter(self, config: WorkflowConfig) -> bool:
        """Check if configuration requires multi-parameter support."""
        
        # Check for multiple strategy parameter combinations
        strategies = self._extract_strategies(config)
        if any(self._has_parameter_grid(strategy) for strategy in strategies):
            return True
            
        # Check for optimization with parameter grids
        if (config.optimization_config and 
            config.optimization_config.get('parameter_grids')):
            return True
            
        # Check for multiple portfolio configurations
        if (hasattr(config, 'parameters') and 
            config.parameters.get('multiple_portfolios')):
            return True
            
        return False
    
    def _extract_strategies(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """Extract all strategies from configuration."""
        strategies = []
        
        # Check top-level strategies
        if hasattr(config, 'strategies') and config.strategies:
            strategies.extend(config.strategies)
        
        # Check backtest config strategies
        if config.backtest_config and config.backtest_config.get('strategies'):
            strategies.extend(config.backtest_config['strategies'])
        
        # Check optimization config strategies
        if config.optimization_config and config.optimization_config.get('strategies'):
            strategies.extend(config.optimization_config['strategies'])
        
        return strategies
    
    def _has_parameter_grid(self, strategy: Dict[str, Any]) -> bool:
        """Check if strategy has parameter grid for optimization."""
        parameters = strategy.get('parameters', {})
        
        # Look for parameter ranges/grids
        for value in parameters.values():
            if isinstance(value, list) and len(value) > 1:
                return True
            if isinstance(value, dict) and ('min' in value and 'max' in value):
                return True
                
        return False
    
    def generate_parameter_combinations(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for multi-parameter execution."""
        
        combinations = []
        
        # Extract strategies with parameter grids
        strategies = self._extract_strategies(config)
        
        for strategy_idx, strategy in enumerate(strategies):
            if self._has_parameter_grid(strategy):
                strategy_combinations = self._generate_strategy_combinations(strategy, strategy_idx)
                combinations.extend(strategy_combinations)
            else:
                # Single parameter set for this strategy
                combinations.append({
                    'strategy_index': strategy_idx,
                    'strategy_type': strategy.get('type', 'unknown'),
                    'parameters': strategy.get('parameters', {}),
                    'combination_id': f"strategy_{strategy_idx}_single"
                })
        
        # Handle optimization parameter grids
        if config.optimization_config and config.optimization_config.get('parameter_grids'):
            opt_combinations = self._generate_optimization_combinations(
                config.optimization_config['parameter_grids']
            )
            # Combine with strategy combinations
            if combinations and opt_combinations:
                # Cross product of strategy and optimization parameters
                combined = []
                for strategy_combo in combinations:
                    for opt_combo in opt_combinations:
                        merged_combo = strategy_combo.copy()
                        merged_combo['optimization_parameters'] = opt_combo['parameters']
                        merged_combo['combination_id'] = f"{strategy_combo['combination_id']}_opt_{opt_combo['combination_id']}"
                        combined.append(merged_combo)
                combinations = combined
            elif opt_combinations:
                combinations = opt_combinations
        
        return combinations
    
    def _generate_strategy_combinations(self, strategy: Dict[str, Any], strategy_idx: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations for a single strategy."""
        
        parameters = strategy.get('parameters', {})
        param_names = []
        param_values = []
        
        for param_name, param_value in parameters.items():
            param_names.append(param_name)
            
            if isinstance(param_value, list):
                param_values.append(param_value)
            elif isinstance(param_value, dict) and 'min' in param_value and 'max' in param_value:
                # Generate range
                min_val = param_value['min']
                max_val = param_value['max']
                step = param_value.get('step', (max_val - min_val) / 10)
                values = []
                current = min_val
                while current <= max_val:
                    values.append(current)
                    current += step
                param_values.append(values)
            else:
                # Single value
                param_values.append([param_value])
        
        # Generate all combinations
        combinations = []
        for combo_idx, value_combination in enumerate(itertools.product(*param_values)):
            param_dict = dict(zip(param_names, value_combination))
            
            combinations.append({
                'strategy_index': strategy_idx,
                'strategy_type': strategy.get('type', 'unknown'),
                'parameters': param_dict,
                'combination_id': f"strategy_{strategy_idx}_combo_{combo_idx}"
            })
        
        logger.info(f"Generated {len(combinations)} parameter combinations for strategy {strategy_idx}")
        return combinations
    
    def _generate_optimization_combinations(self, parameter_grids: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate combinations from optimization parameter grids."""
        
        param_names = []
        param_values = []
        
        for param_name, param_spec in parameter_grids.items():
            param_names.append(param_name)
            
            if isinstance(param_spec, list):
                param_values.append(param_spec)
            elif isinstance(param_spec, dict):
                if 'values' in param_spec:
                    param_values.append(param_spec['values'])
                elif 'min' in param_spec and 'max' in param_spec:
                    # Generate range
                    min_val = param_spec['min']
                    max_val = param_spec['max']
                    step = param_spec.get('step', (max_val - min_val) / 10)
                    values = []
                    current = min_val
                    while current <= max_val:
                        values.append(current)
                        current += step
                    param_values.append(values)
                else:
                    param_values.append([param_spec])  # Single value
            else:
                param_values.append([param_spec])  # Single value
        
        # Generate all combinations
        combinations = []
        for combo_idx, value_combination in enumerate(itertools.product(*param_values)):
            param_dict = dict(zip(param_names, value_combination))
            
            combinations.append({
                'parameters': param_dict,
                'combination_id': f"opt_combo_{combo_idx}"
            })
        
        logger.info(f"Generated {len(combinations)} optimization parameter combinations")
        return combinations
    
    def estimate_execution_complexity(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Estimate execution complexity for resource planning."""
        
        if not self.requires_multi_parameter(config):
            return {
                'complexity': 'simple',
                'parameter_combinations': 1,
                'estimated_containers': 5,  # Typical simple backtest
                'estimated_duration_minutes': 1
            }
        
        combinations = self.generate_parameter_combinations(config)
        num_combinations = len(combinations)
        
        # Estimate resource requirements
        containers_per_combination = 4  # Portfolio + Strategy + Risk + Execution
        shared_containers = 2  # Hub + Data
        total_containers = shared_containers + (containers_per_combination * num_combinations)
        
        # Estimate duration (very rough)
        base_duration = 1  # minutes
        duration_per_combination = 0.5  # minutes
        estimated_duration = base_duration + (duration_per_combination * num_combinations)
        
        if num_combinations <= 10:
            complexity = 'moderate'
        elif num_combinations <= 50:
            complexity = 'high'
        else:
            complexity = 'very_high'
        
        return {
            'complexity': complexity,
            'parameter_combinations': num_combinations,
            'estimated_containers': total_containers,
            'estimated_duration_minutes': estimated_duration,
            'resource_sharing_opportunities': self._analyze_sharing_opportunities(combinations)
        }
    
    def _analyze_sharing_opportunities(self, combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze opportunities for container sharing."""
        
        # Group combinations by shared configuration
        risk_configs = {}
        execution_configs = {}
        
        for combo in combinations:
            # For now, assume sharing opportunities exist
            # This would be expanded to analyze actual configuration similarity
            risk_key = "default_risk"  # Would hash actual risk config
            execution_key = "default_execution"  # Would hash actual execution config
            
            if risk_key not in risk_configs:
                risk_configs[risk_key] = []
            risk_configs[risk_key].append(combo['combination_id'])
            
            if execution_key not in execution_configs:
                execution_configs[execution_key] = []
            execution_configs[execution_key].append(combo['combination_id'])
        
        return {
            'shared_risk_containers': len(risk_configs),
            'shared_execution_containers': len(execution_configs),
            'potential_savings_percent': max(0, (len(combinations) - len(risk_configs) - len(execution_configs)) / len(combinations) * 100)
        }