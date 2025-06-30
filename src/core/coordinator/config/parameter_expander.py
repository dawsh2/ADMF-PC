"""
Parameter Space Expansion for Optimization

Handles expansion of parameter_space configuration section for optimization workflows.
Supports wildcard expansion, category-based selection, and explicit strategy lists.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from itertools import product

from ...components.discovery import get_component_registry
from ..feature_discovery import FeatureDiscovery

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpec:
    """Specification for an optimizable parameter."""
    param_type: str  # 'int', 'float', 'categorical'
    default: Any
    range: Optional[tuple] = None  # For numeric types (min, max)
    choices: Optional[List[Any]] = None  # For categorical
    step: Optional[float] = None  # For numeric types
    description: Optional[str] = None


class ParameterSpaceExpander:
    """
    Expands parameter_space configuration into concrete parameter combinations.
    
    Supports:
    - Wildcard expansion: "indicators: *" 
    - Category expansion: "indicators: {crossover: *, momentum: ['macd', 'rsi']}"
    - Explicit lists: "strategies: ['sma_crossover', 'ema_crossover']"
    - Parameter overrides: Per-strategy parameter customization
    - Recursive ensemble expansion
    """
    
    def __init__(self, granularity: int = 5):
        self.registry = get_component_registry()
        self.feature_discovery = FeatureDiscovery()
        self.granularity = granularity  # Default number of samples for ranges
        
    def expand_parameter_space(self, config: Dict[str, Any], 
                             optimize: bool = False) -> Dict[str, Any]:
        """
        Expand parameter_space section when arrays are present or --optimize flag is set.
        
        Args:
            config: Full configuration dict
            optimize: Whether optimization mode is active
            
        Returns:
            Expanded configuration with all parameter combinations
        """
        # Check if we need to expand - either optimize flag is set, or arrays exist in config
        needs_expansion = optimize
        
        if not needs_expansion and 'parameter_space' in config:
            # Check if any strategies have array parameters
            strategies = config.get('parameter_space', {}).get('strategies', [])
            for strategy in strategies:
                param_overrides = strategy.get('param_overrides', {})
                if any(isinstance(v, list) for v in param_overrides.values()):
                    needs_expansion = True
                    break
                    
        if not needs_expansion:
            # No expansion needed
            return config
            
        if 'parameter_space' not in config:
            logger.info("No parameter_space section found, extracting from strategy section")
            # Extract parameter space from strategy configuration
            extracted_param_space = self._extract_parameter_space_from_strategy(config.get('strategy', {}))
            
            if extracted_param_space:
                logger.info(f"Extracted parameter space for {len(extracted_param_space)} strategies")
                config['parameter_space'] = {'strategies': extracted_param_space}
                param_space = config['parameter_space']
            else:
                logger.warning("Could not extract any strategies from config, using as-is")
                return config
        else:
            param_space = config['parameter_space']
        
        # Expand indicators section
        if 'indicators' in param_space:
            expanded_indicators = self._expand_indicators(param_space['indicators'])
        else:
            expanded_indicators = []
            
        # Expand classifiers section  
        if 'classifiers' in param_space:
            expanded_classifiers = self._expand_classifiers(param_space['classifiers'])
        else:
            expanded_classifiers = []
            
        # Expand custom strategies section
        if 'strategies' in param_space:
            expanded_strategies = self._expand_strategies(param_space['strategies'])
        else:
            expanded_strategies = []
            
        # Build expanded config
        expanded_config = config.copy()
        expanded_config['expanded_strategies'] = (
            expanded_indicators + expanded_classifiers + expanded_strategies
        )
        
        # Generate parameter combinations
        expanded_config['parameter_combinations'] = self._generate_combinations(
            expanded_config['expanded_strategies']
        )
        
        logger.info(f"Expanded {len(expanded_config['parameter_combinations'])} parameter combinations")
        
        return expanded_config
    
    def _expand_indicators(self, indicators_spec: Union[str, Dict, List]) -> List[Dict[str, Any]]:
        """Expand indicators specification."""
        if indicators_spec == "*":
            # Expand all indicators with default parameters
            return self._get_all_indicators()
            
        elif isinstance(indicators_spec, dict):
            # Check for exclude list first
            exclude_list = []
            if 'exclude' in indicators_spec:
                exclude_list = indicators_spec.get('exclude', [])
                if isinstance(exclude_list, str):
                    exclude_list = [exclude_list]
                logger.info(f"Excluding strategies: {exclude_list}")
            
            # Category-based expansion
            expanded = []
            for category, spec in indicators_spec.items():
                if category == 'exclude':
                    continue  # Skip the exclude key
                    
                if spec == "*":
                    expanded.extend(self._get_indicators_by_category(category))
                elif isinstance(spec, list):
                    for strategy_name in spec:
                        expanded.append(self._get_strategy_config(strategy_name))
                else:
                    logger.warning(f"Unknown indicator spec format: {spec}")
            
            # Apply exclusions
            if exclude_list:
                expanded = [s for s in expanded if s.get('name') not in exclude_list]
                logger.info(f"After exclusions: {len(expanded)} strategies remaining")
                
            return expanded
            
        elif isinstance(indicators_spec, list):
            # Explicit list
            return [self._get_strategy_config(name) for name in indicators_spec]
            
        else:
            logger.warning(f"Unknown indicators specification: {indicators_spec}")
            return []
    
    def _expand_classifiers(self, classifiers_spec: Union[str, List]) -> List[Dict[str, Any]]:
        """Expand classifiers specification."""
        if classifiers_spec == "*":
            # Get all classifiers
            return self._get_all_classifiers()
            
        elif isinstance(classifiers_spec, list):
            # Explicit list
            return [self._get_strategy_config(name, 'classifier') for name in classifiers_spec]
            
        else:
            logger.warning(f"Unknown classifiers specification: {classifiers_spec}")
            return []
    
    def _expand_strategies(self, strategies_spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand custom strategies with parameter overrides."""
        expanded = []
        
        for strategy in strategies_spec:
            strategy_type = strategy.get('type')
            if not strategy_type:
                logger.warning(f"Strategy missing type: {strategy}")
                continue
                
            base_config = self._get_strategy_config(strategy_type)
            
            # If strategy not found in registry, use the original config
            if not base_config:
                base_config = {
                    'type': strategy_type,
                    'name': strategy.get('name', strategy_type),
                    'parameter_space': {}
                }
            
            # Apply parameter overrides
            if 'param_overrides' in strategy:
                base_config['param_overrides'] = strategy['param_overrides']
                
            # Preserve constraints/threshold/filter if present
            if 'constraints' in strategy:
                base_config['constraints'] = strategy['constraints']
            elif 'threshold' in strategy:
                base_config['threshold'] = strategy['threshold']
            elif 'filter' in strategy:
                base_config['filter'] = strategy['filter']
                
            # Handle ensemble recursively
            if strategy_type == 'universal_ensemble' and 'strategies' in strategy:
                base_config['sub_strategies'] = self._expand_strategies(strategy['strategies'])
                
            expanded.append(base_config)
            
        return expanded
    
    def _get_all_indicators(self) -> List[Dict[str, Any]]:
        """Get all registered indicator strategies."""
        strategies = []
        
        # Get all strategy components
        for component_id, info in self.registry._components.items():
            if info.component_type == 'strategy':
                # Check if it's from the indicators module
                module_name = info.metadata.get('module', '')
                if 'indicators' in module_name:
                    strategies.append({
                        'type': component_id,
                        'name': component_id,
                        'parameter_space': info.metadata.get('parameter_space', {})
                    })
                # Also check for category metadata
                elif 'indicators' in info.metadata.get('categories', []):
                    strategies.append({
                        'type': component_id,
                        'name': component_id,
                        'parameter_space': info.metadata.get('parameter_space', {})
                    })
                    
        logger.info(f"Found {len(strategies)} indicator strategies")
        return strategies
    
    def _get_indicators_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get indicators by category (crossover, momentum, etc)."""
        strategies = []
        
        # Check if category is a strategy type (mean_reversion, trend_following, etc.)
        strategy_types = ['mean_reversion', 'trend_following', 'breakout', 'oscillator', 'structure']
        if category in strategy_types:
            # Filter by strategy_type metadata
            for component_id, info in self.registry._components.items():
                if (info.component_type == 'strategy' and 
                    info.metadata.get('strategy_type') == category):
                    strategies.append({
                        'type': component_id,
                        'name': component_id,
                        'parameter_space': info.metadata.get('parameter_space', {})
                    })
            return strategies
        
        # Check if category is in tags
        for component_id, info in self.registry._components.items():
            if (info.component_type == 'strategy' and 
                category in info.metadata.get('tags', [])):
                strategies.append({
                    'type': component_id,
                    'name': component_id,
                    'parameter_space': info.metadata.get('parameter_space', {})
                })
        
        # Fallback to pattern matching
        category_patterns = {
            'crossover': ['_crossover', '_cross'],
            'momentum': ['momentum', 'rsi', 'macd', 'roc'],
            'oscillator': ['rsi', 'cci', 'stochastic', 'williams'],
            'trend': ['adx', 'aroon', 'supertrend', 'sar'],
            'volatility': ['atr', 'bollinger', 'keltner'],
            'volume': ['obv', 'vwap', 'chaikin', 'mfi'],
            'structure': ['pivot', 'support', 'trendline']
        }
        
        patterns = category_patterns.get(category, [category])
        
        for component_id, info in self.registry._components.items():
            if info.component_type == 'strategy':
                # Check if strategy matches category pattern
                if any(pattern in component_id.lower() for pattern in patterns):
                    # Avoid duplicates
                    if not any(s['name'] == component_id for s in strategies):
                        strategies.append({
                            'type': component_id,
                            'name': component_id,
                            'parameter_space': info.metadata.get('parameter_space', {})
                        })
                    
        return strategies
    
    def _get_all_classifiers(self) -> List[Dict[str, Any]]:
        """Get all registered classifiers."""
        classifiers = []
        
        for component_id, info in self.registry._components.items():
            if info.component_type == 'classifier':
                classifiers.append({
                    'type': component_id,
                    'name': component_id,
                    'parameter_space': info.metadata.get('parameter_space', {})
                })
                
        return classifiers
    
    def _get_strategy_config(self, strategy_name: str, 
                           component_type: str = 'strategy') -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        info = self.registry.get_component(strategy_name)
        
        if not info:
            logger.warning(f"Strategy not found: {strategy_name}")
            return {}
            
        # Verify it's the right component type
        if info.component_type != component_type:
            logger.warning(f"Component {strategy_name} is type {info.component_type}, not {component_type}")
            return {}
            
        return {
            'type': strategy_name,
            'name': strategy_name,
            'parameter_space': info.metadata.get('parameter_space', {})
        }
    
    def _generate_combinations(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for optimization.
        
        For each strategy, expand its parameter space into concrete combinations.
        """
        all_combinations = []
        
        for strategy in strategies:
            param_space = strategy.get('parameter_space', {})
            param_overrides = strategy.get('param_overrides', {})
            
            # Generate combinations for this strategy
            combinations = self._expand_strategy_parameters(param_space, param_overrides)
            
            for params in combinations:
                combo = {
                    'strategy_type': strategy['type'],
                    'strategy_name': strategy['name'],
                    'parameters': params
                }
                
                # Pass through constraints/threshold/filter if present
                if 'constraints' in strategy:
                    combo['constraints'] = strategy['constraints']
                elif 'threshold' in strategy:
                    combo['threshold'] = strategy['threshold']
                elif 'filter' in strategy:
                    combo['filter'] = strategy['filter']
                
                all_combinations.append(combo)
                
        return all_combinations
    
    def _expand_strategy_parameters(self, param_space: Dict[str, Any], 
                                  overrides: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand parameter space for a single strategy.
        
        Returns list of parameter dictionaries representing all combinations.
        """
        if not param_space and not overrides:
            return [{}]  # Single combination with no parameters
            
        # Apply overrides to parameter space
        effective_space = param_space.copy()
        for param, override in overrides.items():
            if isinstance(override, list):
                # Explicit values
                effective_space[param] = {'values': override}
            elif isinstance(override, dict):
                # Override specification
                effective_space[param].update(override)
            else:
                # Single value
                effective_space[param] = {'values': [override]}
                
        # Generate parameter grid
        param_grid = {}
        for param_name, spec in effective_space.items():
            if isinstance(spec, dict):
                if 'values' in spec:
                    # Explicit values provided
                    param_grid[param_name] = spec['values']
                elif spec.get('type') == 'bool':
                    # Boolean parameters always test both values
                    param_grid[param_name] = [True, False]
                elif 'range' in spec and 'type' in spec:
                    # Generate values from range
                    param_grid[param_name] = self._sample_parameter_range(spec, self.granularity)
                elif 'default' in spec:
                    # Use default only
                    param_grid[param_name] = [spec['default']]
                else:
                    logger.warning(f"Cannot expand parameter {param_name}: {spec}")
            else:
                # Assume it's a list of values
                param_grid[param_name] = spec if isinstance(spec, list) else [spec]
                
        # Generate all combinations
        if not param_grid:
            return [{}]
            
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        return combinations
    
    def _sample_parameter_range(self, spec: Dict[str, Any], 
                               num_samples: int = 5) -> List[Any]:
        """Sample values from a parameter range specification."""
        param_type = spec.get('type', 'float')
        param_range = spec.get('range', [])
        step = spec.get('step')  # Optional step parameter
        
        # Check for explicit values first
        if 'values' in spec and spec['values']:
            return spec['values']
        
        # Use parameter-specific granularity if defined
        param_granularity = spec.get('granularity', num_samples)
        
        if len(param_range) != 2:
            logger.warning(f"Invalid range specification: {spec}")
            return [spec.get('default', 0)]
            
        min_val, max_val = param_range
        
        if param_type == 'int':
            # Integer sampling
            if step:
                # Use explicit step
                return list(range(min_val, max_val + 1, step))
            elif max_val - min_val + 1 <= param_granularity:
                # Return all values in range
                return list(range(min_val, max_val + 1))
            else:
                # Sample evenly with granularity
                import numpy as np
                values = np.linspace(min_val, max_val, param_granularity)
                # Convert to integers and remove duplicates
                int_values = [int(round(v)) for v in values]
                return sorted(list(set(int_values)))
                
        elif param_type == 'float':
            # Float sampling
            if step:
                # Use explicit step
                values = []
                val = min_val
                while val <= max_val:
                    values.append(round(val, 6))
                    val += step
                return values
            else:
                # Sample evenly with granularity
                import numpy as np
                values = np.linspace(min_val, max_val, param_granularity)
                return [round(float(v), 4) for v in values]
            
        else:
            logger.warning(f"Unknown parameter type: {param_type}")
            return [spec.get('default', min_val)]


class EnsembleParameterExtractor:
    """
    Recursively extracts parameter spaces from ensemble configurations.
    
    Handles nested ensembles and maintains parameter namespacing to avoid conflicts.
    """
    
    def __init__(self):
        self.registry = get_component_registry()
        
    def extract_parameters(self, ensemble_config: Dict[str, Any], 
                         prefix: str = "") -> Dict[str, Any]:
        """
        Recursively extract all parameters from ensemble and sub-strategies.
        
        Args:
            ensemble_config: Ensemble strategy configuration
            prefix: Namespace prefix for parameters
            
        Returns:
            Flattened parameter space with namespaced parameter names
        """
        all_params = {}
        
        if ensemble_config.get('type') == 'universal_ensemble' or 'strategies' in ensemble_config:
            # Process each sub-strategy
            for idx, sub_strategy in enumerate(ensemble_config.get('strategies', [])):
                strategy_name = sub_strategy.get('name', f'strategy_{idx}')
                param_prefix = f"{prefix}{strategy_name}." if prefix else f"{strategy_name}."
                
                if sub_strategy.get('type') == 'universal_ensemble':
                    # Recursive extraction for nested ensemble
                    sub_params = self.extract_parameters(sub_strategy, param_prefix)
                    all_params.update(sub_params)
                else:
                    # Extract parameters for atomic strategy
                    strategy_params = self._get_strategy_parameters(sub_strategy)
                    
                    # Namespace the parameters
                    for param_name, param_spec in strategy_params.items():
                        all_params[f"{param_prefix}{param_name}"] = param_spec
                        
        else:
            # Single strategy - extract its parameters
            strategy_params = self._get_strategy_parameters(ensemble_config)
            
            for param_name, param_spec in strategy_params.items():
                full_name = f"{prefix}{param_name}" if prefix else param_name
                all_params[full_name] = param_spec
                
        return all_params
    
    def _extract_parameter_space_from_strategy(self, strategy_config: Any) -> List[Dict[str, Any]]:
        """
        Extract parameter space from a strategy configuration.
        
        This allows --optimize to work even without explicit parameter_space section.
        """
        extracted = []
        
        if isinstance(strategy_config, list):
            # Composite strategy - extract from each component
            for item in strategy_config:
                extracted.extend(self._extract_parameter_space_from_strategy(item))
                
        elif isinstance(strategy_config, dict):
            # Skip known non-strategy keys
            skip_keys = {'weight', 'condition', 'combination', 'weight_threshold', 
                        'if_true', 'if_false', 'strategy', 'strategies', 'regime', 
                        'cases', 'regimes', 'conditions'}
            
            # Check for nested strategies
            if 'strategy' in strategy_config:
                extracted.extend(self._extract_parameter_space_from_strategy(strategy_config['strategy']))
            elif 'strategies' in strategy_config:
                extracted.extend(self._extract_parameter_space_from_strategy(strategy_config['strategies']))
            else:
                # Look for atomic strategy
                for key, value in strategy_config.items():
                    if key not in skip_keys:
                        # This might be a strategy type
                        component_info = self.registry.get_component(key)
                        if component_info and component_info.component_type == 'strategy':
                            # Extract strategy config
                            strategy_params = value.get('params', {}) if isinstance(value, dict) else {}
                            
                            # Get default parameter space from registry
                            default_param_space = component_info.metadata.get('parameter_space', {})
                            
                            # Create extraction entry
                            entry = {
                                'type': key,
                                'name': key,
                                'params': strategy_params,
                                'parameter_space': default_param_space
                            }
                            
                            # Extract filter if present
                            if isinstance(value, dict) and 'filter' in value:
                                entry['filter'] = value['filter']
                            
                            extracted.append(entry)
                            break
        
        return extracted
    
    def _get_strategy_parameters(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameter space for a single strategy."""
        strategy_type = strategy_config.get('type')
        
        if not strategy_type:
            return {}
            
        # Get from registry
        info = self.registry.get_component(strategy_type, 'strategy')
        
        if not info:
            logger.warning(f"Strategy {strategy_type} not found in registry")
            return {}
            
        # Get default parameter space
        default_space = info.metadata.get('parameter_space', {})
        
        # Apply any configured parameters as constraints
        configured_params = strategy_config.get('params', {})
        
        constrained_space = {}
        for param_name, param_spec in default_space.items():
            if param_name in configured_params:
                # Parameter is fixed by configuration
                constrained_space[param_name] = {
                    'type': param_spec.get('type', 'float'),
                    'values': [configured_params[param_name]],
                    'default': configured_params[param_name]
                }
            else:
                # Use full parameter space
                constrained_space[param_name] = param_spec
                
        return constrained_space