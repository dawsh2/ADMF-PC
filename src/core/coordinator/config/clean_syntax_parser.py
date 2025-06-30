"""
Clean Config Syntax Parser

Transforms the new clean YAML syntax into the internal format.

Example input:
```yaml
strategy:
  - keltner_bands:
      period: [10, 20, 30]
      multiplier: [1.5, 2.0]
      filter: [
        null,
        {rsi_below: {threshold: 50}},
        {volume_above: {multiplier: 1.2}}
      ]
```

Transforms to internal parameter_space format.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from copy import deepcopy

logger = logging.getLogger(__name__)


class CleanSyntaxParser:
    """Parse clean config syntax and expand to internal format."""
    
    # Built-in filter types and their expansions
    FILTER_TEMPLATES = {
        'rsi_below': 'rsi_{period} < {threshold}',
        'rsi_above': 'rsi_{period} > {threshold}',
        'volume_above': 'volume > volume_sma_{sma_period} * {multiplier}',
        'volume_below': 'volume < volume_sma_{sma_period} * {multiplier}',
        'volatility_above': 'atr_{atr_period} > atr_sma_{atr_sma_period} * {threshold}',
        'volatility_below': 'atr_{atr_period} < atr_sma_{atr_sma_period} * {threshold}',
        'volatility_range': '(atr_{atr_period} / atr_sma_{atr_sma_period} >= {min} and atr_{atr_period} / atr_sma_{atr_sma_period} <= {max})',
        'price_below_vwap': 'close < vwap * {factor}',
        'price_above_vwap': 'close > vwap * {factor}',
        'price_distance_vwap': 'abs(close - vwap) / vwap > {min}',
        'time_exclude': '(bar_of_day < {start_bar} or bar_of_day > {end_bar})',
        'time_include': '(bar_of_day >= {start_bar} and bar_of_day <= {end_bar})',
        'atr_ratio_above': 'atr_{period} / atr_{baseline} > {threshold}',
        'trend_strength_below': 'abs(sma_{fast_period} - sma_{slow_period}) / sma_{slow_period} < {threshold}',
    }
    
    # Default parameters for filters
    FILTER_DEFAULTS = {
        'rsi_below': {'period': 14},
        'rsi_above': {'period': 14},
        'volume_above': {'sma_period': 20},
        'volume_below': {'sma_period': 20},
        'volatility_above': {'atr_period': 14, 'atr_sma_period': 50},
        'volatility_below': {'atr_period': 14, 'atr_sma_period': 50},
        'volatility_range': {'atr_period': 14, 'atr_sma_period': 50},
        'price_below_vwap': {'factor': 1.0},
        'price_above_vwap': {'factor': 1.0},
        'price_distance_vwap': {'min': 0.001},
        'time_exclude': {},
        'time_include': {},
        'atr_ratio_above': {'period': 14, 'baseline': 50},
        'trend_strength_below': {'fast_period': 10, 'slow_period': 50},
    }
    
    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse clean syntax config and return internal format.
        
        Args:
            config: Clean syntax config dict
            
        Returns:
            Internal format config dict
        """
        result = deepcopy(config)
        logger.debug(f"Clean syntax parser - input config has data: {config.get('data')}")
        
        # Handle strategy field if present
        if 'strategy' in config:
            parsed = self._parse_strategy_field(config['strategy'])
            # If it returned a strategy field, keep it as strategy (for ensemble)
            if 'strategy' in parsed:
                result['strategy'] = parsed['strategy']
            # Otherwise it's a parameter space (for optimization)
            elif 'parameter_space' in parsed:
                result['parameter_space'] = parsed['parameter_space']
            else:
                result.update(parsed)
            # Only remove the original strategy field if we created parameter_space
            if 'parameter_space' in result and 'strategy' in result:
                del result['strategy']
        
        # Handle symbols/timeframe shorthand
        if 'symbols' in config and isinstance(config['symbols'], list):
            # Already in correct format
            pass
        elif 'symbol' in config:
            # Single symbol shorthand
            result['symbols'] = [config['symbol']]
            del result['symbol']
            
        if 'timeframe' in config and 'timeframes' not in config:
            result['timeframes'] = [config['timeframe']]
            del result['timeframe']
            
        logger.debug(f"Clean syntax parser - output config has data: {result.get('data')}")
        return result
    
    def _parse_strategy_field(self, strategies: List[Union[Dict, str]]) -> Dict[str, Any]:
        """
        Parse the strategy field into parameter_space format.
        
        Handles three cases:
        1. Single strategy with parameter expansion
        2. Multiple strategies without parameter expansion (pure ensemble)
        3. Multiple strategies with parameter expansion (ensemble of expanded strategies)
        
        Args:
            strategies: List of strategy definitions
            
        Returns:
            parameter_space dict
        """
        # First, determine what we're dealing with
        has_parameter_lists = self._has_parameter_lists(strategies)
        is_multiple_strategies = len(strategies) > 1
        
        if not has_parameter_lists and is_multiple_strategies:
            # Pure ensemble - return as-is for compiler to handle
            return {'strategy': strategies}
        elif has_parameter_lists and is_multiple_strategies:
            # Ensemble with parameter expansion
            # TODO: Add --flatten flag support to run individually
            return self._create_ensemble_parameter_space(strategies)
        else:
            # Single strategy parameter expansion (or simple single strategy)
            parameter_space = {'strategies': []}
            
            for strategy_def in strategies:
                if isinstance(strategy_def, dict):
                    # Check if this dict contains both a strategy and a constraint/threshold
                    constraint = strategy_def.get('constraints') or strategy_def.get('threshold')
                    risk_params = strategy_def.get('risk')
                    
                    # Parse each strategy type (skipping 'constraints', 'threshold', and 'risk' keys)
                    for strategy_type, params in strategy_def.items():
                        if strategy_type in ['constraints', 'threshold', 'risk']:
                            continue
                        expanded_strategies = self._expand_strategy(strategy_type, params)
                        
                        # If there's a constraint at this level, add it to each expanded strategy
                        if constraint:
                            for strategy in expanded_strategies:
                                # Use 'constraints' for new configs, keep 'threshold' for backward compat
                                if 'constraints' in strategy_def:
                                    strategy['constraints'] = constraint
                                else:
                                    strategy['threshold'] = constraint
                        
                        # If there's risk config at this level, add it to each expanded strategy
                        if risk_params:
                            for strategy in expanded_strategies:
                                if 'param_overrides' not in strategy:
                                    strategy['param_overrides'] = {}
                                strategy['param_overrides']['_risk'] = risk_params
                        
                        parameter_space['strategies'].extend(expanded_strategies)
                else:
                    # Simple string strategy name
                    parameter_space['strategies'].append({
                        'type': strategy_def,
                        'param_overrides': {}
                    })
                    
            return {'parameter_space': parameter_space}
    
    def _expand_strategy(self, strategy_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand a single strategy definition with filters.
        
        Args:
            strategy_type: Name of the strategy
            params: Parameters including filter specifications
            
        Returns:
            List of expanded strategy definitions
        """
        # Check for paired parameters syntax
        if 'params' in params and isinstance(params['params'], list):
            # Handle paired parameters
            expanded = []
            for param_set in params['params']:
                expanded.append({
                    'type': strategy_type,
                    'name': strategy_type,
                    'param_overrides': param_set
                })
            return expanded
        
        # Extract filter if present
        filters = params.pop('filter', [None])
        if not isinstance(filters, list):
            filters = [filters]
        
        # Extract constraints/threshold if present
        constraints = params.pop('constraints', params.pop('threshold', None))
        
        # Base parameters (everything except filter)
        base_params = {}
        param_lists = {}  # Track which params have multiple values
        for k, v in params.items():
            if k != 'filter':
                # Check if it's a range() expression
                if isinstance(v, str) and v.startswith('range('):
                    values = self._parse_range(v)
                    base_params[k] = values
                    if len(values) > 1:
                        param_lists[k] = values
                elif isinstance(v, list) and len(v) > 1:
                    base_params[k] = v
                    param_lists[k] = v
                else:
                    base_params[k] = v
        
        # If we have parameter lists, we need to expand them before dealing with filters
        if param_lists:
            # Generate all parameter combinations
            import itertools
            param_names = list(param_lists.keys())
            param_values = [param_lists[name] for name in param_names]
            
            base_combinations = []
            for combo in itertools.product(*param_values):
                combo_params = {}
                # Add scalar params
                for k, v in base_params.items():
                    if k not in param_lists:
                        combo_params[k] = v
                # Add combination values
                for i, param_name in enumerate(param_names):
                    combo_params[param_name] = combo[i]
                
                # Generate unique name for this combination
                param_parts = [str(v).replace('.', '') for v in combo]
                combo_name = f"{strategy_type}_{'_'.join(param_parts)}"
                
                base_combinations.append({
                    'params': combo_params,
                    'name': combo_name
                })
        else:
            # No parameter expansion needed
            base_combinations = [{
                'params': base_params,
                'name': strategy_type
            }]
        
        # Now apply filters to each base combination
        expanded = []
        
        # If no filters, return the base combinations
        if not filters or filters == [None]:
            for base_combo in base_combinations:
                strategy_def = {
                    'type': strategy_type,
                    'name': base_combo['name'],
                    'param_overrides': base_combo['params']
                }
                # Add constraints/threshold if present
                if constraints:
                    strategy_def['constraints'] = constraints
                expanded.append(strategy_def)
            return expanded
        
        # Expand each filter variant for each base combination
        for base_combo in base_combinations:
            for i, filter_spec in enumerate(filters):
                if filter_spec is None:
                    # No filter variant
                    strategy_def = {
                        'type': strategy_type,
                        'name': base_combo['name'] if len(filters) == 1 else f"{base_combo['name']}_f{i}",
                        'param_overrides': base_combo['params'].copy()
                    }
                    # Add constraints/threshold if present
                    if constraints:
                        strategy_def['constraints'] = constraints
                    expanded.append(strategy_def)
                else:
                    # Parse and add filter
                    filter_expr, filter_params = self._parse_filter(filter_spec)
                    
                    # If filter has parameter sweeps, expand them
                    if filter_params and any(isinstance(v, list) for v in filter_params.values()):
                        # Expand filter parameter combinations
                        param_combinations = self._expand_filter_params(filter_params)
                        
                        for j, param_combo in enumerate(param_combinations):
                            # Replace placeholders in filter expression
                            expanded_filter = filter_expr
                            for param_name, param_value in param_combo.items():
                                placeholder = f"${{{param_name}}}"
                                expanded_filter = expanded_filter.replace(placeholder, str(param_value))
                            
                            # Generate unique name including filter index
                            filter_suffix = f"_f{i}" if len(filters) > 1 else ""
                            param_suffix = f"_p{j}" if len(param_combinations) > 1 else ""
                            strategy_name = f"{base_combo['name']}{filter_suffix}{param_suffix}"
                            
                            strategy_def = {
                                'type': strategy_type,
                                'name': strategy_name,
                                'param_overrides': base_combo['params'].copy(),
                                'filter': f"signal == 0 or ({expanded_filter})"
                            }
                            # Add constraints/threshold if present
                            if constraints:
                                strategy_def['constraints'] = constraints
                            expanded.append(strategy_def)
                    else:
                        # No parameter sweeps in filter
                        filter_suffix = f"_f{i}" if len(filters) > 1 else ""
                        strategy_def = {
                            'type': strategy_type,
                            'name': f"{base_combo['name']}{filter_suffix}",
                            'param_overrides': base_combo['params'].copy()
                        }
                        
                        if filter_expr:
                            strategy_def['filter'] = f"signal == 0 or ({filter_expr})"
                        
                        # Add constraints/threshold if present
                        if constraints:
                            strategy_def['constraints'] = constraints
                            
                        expanded.append(strategy_def)
                
        return expanded
    
    def _expand_filter_params(self, filter_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand filter parameters with arrays into all combinations.
        
        Args:
            filter_params: Dict with some values as lists
            
        Returns:
            List of dicts with all parameter combinations
        """
        import itertools
        
        # Separate array params from scalar params
        array_params = {}
        scalar_params = {}
        
        for key, value in filter_params.items():
            if isinstance(value, list):
                array_params[key] = value
            else:
                scalar_params[key] = value
        
        if not array_params:
            # No arrays to expand
            return [filter_params]
        
        # Generate all combinations
        param_names = list(array_params.keys())
        param_values = list(array_params.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combo_dict = scalar_params.copy()
            for i, param_name in enumerate(param_names):
                combo_dict[param_name] = combo[i]
            combinations.append(combo_dict)
        
        return combinations
    
    def _parse_range(self, range_str: str) -> List[Union[int, float]]:
        """
        Parse a range(start, stop, step) expression into a list of values.
        
        Args:
            range_str: String like "range(10, 50, 1)" or "range(0.5, 4.0, 0.2)"
            
        Returns:
            List of values
        """
        import re
        import numpy as np
        
        # Extract numbers from range() expression
        match = re.match(r'range\(([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\)', range_str.strip())
        if not match:
            logger.warning(f"Invalid range expression: {range_str}")
            return []
        
        start = float(match.group(1))
        stop = float(match.group(2))
        step = float(match.group(3))
        
        # Check if all values are integers
        if all(x == int(x) for x in [start, stop, step]):
            # Integer range
            return list(range(int(start), int(stop) + 1, int(step)))
        else:
            # Float range - use numpy arange for better precision
            import numpy as np
            # Use numpy's arange and include endpoint
            values = np.arange(start, stop + step/2, step)
            # Round to avoid floating point display issues
            return [round(float(v), 6) for v in values]
    
    def _parse_filter(self, filter_spec: Union[Dict, List, str, bool]) -> tuple[str, Dict[str, Any]]:
        """
        Parse a filter specification into expression and parameters.
        
        Args:
            filter_spec: Filter specification
            
        Returns:
            Tuple of (filter_expression, filter_parameters)
        """
        if isinstance(filter_spec, bool):
            # false means "block all signals"
            if not filter_spec:
                return "False", {}
            else:
                return "", {}
                
        elif isinstance(filter_spec, str):
            # Raw filter expression
            return filter_spec, {}
            
        elif isinstance(filter_spec, list):
            # Array of filters (AND logic) - needs special handling for parameter expansion
            return self._parse_combined_filter(filter_spec)
            
        elif isinstance(filter_spec, dict):
            # Check for directional filter
            if 'long' in filter_spec or 'short' in filter_spec:
                return self._parse_directional_filter(filter_spec)
            else:
                # Single filter dict
                return self._parse_single_filter(filter_spec)
                
        return "", {}
    
    def _parse_combined_filter(self, filter_list: List[Any]) -> tuple[str, Dict[str, Any]]:
        """
        Parse combined filters (AND logic) with proper parameter expansion.
        
        This method handles the case where multiple filters each have parameter sweeps.
        For example: [{rsi_below: {threshold: [50, 60]}}, {volume_above: {multiplier: [1.1, 1.2]}}]
        Should expand to 2x2=4 combinations.
        """
        expressions = []
        all_params = {}
        
        # First pass: collect all expressions and parameters
        for sub_filter in filter_list:
            expr, params = self._parse_filter(sub_filter)
            if expr:
                expressions.append(expr)
            all_params.update(params)
        
        # Join expressions with AND
        combined_expr = " and ".join(f"({expr})" for expr in expressions if expr)
        
        return combined_expr, all_params
    
    def _parse_directional_filter(self, filter_spec: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Parse directional filter with long/short conditions."""
        expressions = []
        all_params = {}
        
        # Parse long conditions
        if 'long' in filter_spec:
            long_spec = filter_spec['long']
            if isinstance(long_spec, bool) and not long_spec:
                # long: false means no long signals
                expressions.append("signal <= 0")
            else:
                long_expr, long_params = self._parse_filter(long_spec)
                if long_expr:
                    # For directional filters, we need to combine conditions properly
                    # Long signals should pass when signal > 0 AND the condition is met
                    expressions.append(f"(signal > 0 and ({long_expr}))")
                    all_params.update(long_params)
        
        # Parse short conditions
        if 'short' in filter_spec:
            short_spec = filter_spec['short']
            if isinstance(short_spec, bool) and not short_spec:
                # short: false means no short signals
                expressions.append("signal >= 0")
            else:
                short_expr, short_params = self._parse_filter(short_spec)
                if short_expr:
                    # Short signals should pass when signal < 0 AND the condition is met
                    expressions.append(f"(signal < 0 and ({short_expr}))")
                    all_params.update(short_params)
        
        # For directional filters, we use OR logic because we want to allow
        # either long signals (when conditions met) OR short signals (when conditions met)
        # OR neutral signals (signal == 0)
        if expressions:
            # Add neutral signal passthrough
            combined = " or ".join(expressions)
            return f"signal == 0 or ({combined})", all_params
        else:
            return "", all_params
    
    def _parse_single_filter(self, filter_spec: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Parse a single filter specification."""
        # Should have exactly one key (the filter type)
        if len(filter_spec) != 1:
            return "", {}
            
        filter_type, filter_config = next(iter(filter_spec.items()))
        
        if filter_type not in self.FILTER_TEMPLATES:
            logger.warning(f"Unknown filter type: {filter_type}")
            return "", {}
            
        # Get template and defaults
        template = self.FILTER_TEMPLATES[filter_type]
        defaults = self.FILTER_DEFAULTS.get(filter_type, {})
        
        # Merge config with defaults
        params = defaults.copy()
        if isinstance(filter_config, dict):
            params.update(filter_config)
            
        # Handle special cases
        if filter_type in ['time_exclude', 'time_include']:
            params = self._parse_time_filter(params)
        elif filter_type in ['price_below_vwap', 'price_above_vwap'] and 'buffer' in params:
            # Convert buffer to factor
            buffer = params.pop('buffer')
            if isinstance(buffer, list):
                # Handle parameter sweep
                if filter_type == 'price_below_vwap':
                    params['factor'] = [1.0 - b for b in buffer]
                else:
                    params['factor'] = [1.0 + b for b in buffer]
            else:
                # Single value
                if filter_type == 'price_below_vwap':
                    params['factor'] = 1.0 - buffer
                else:
                    params['factor'] = 1.0 + buffer
                
        # Extract parameter sweeps
        filter_params = {}
        template_params = params.copy()
        
        for key, value in params.items():
            if isinstance(value, list):
                # This is a parameter sweep
                param_name = f"{filter_type}_{key}"
                filter_params[param_name] = value
                template_params[key] = f"${{{param_name}}}"
                
        # Format the template
        try:
            expression = template.format(**template_params)
        except KeyError as e:
            logger.error(f"Missing parameter for filter {filter_type}: {e}")
            return "", {}
            
        return expression, filter_params
    
    def _parse_time_filter(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert time strings to bar numbers."""
        result = params.copy()
        
        # Convert time strings to bar numbers (assuming 5min bars, market hours 9:30-16:00)
        if 'start' in params:
            result['start_bar'] = self._time_to_bar(params['start'])
        if 'end' in params:
            result['end_bar'] = self._time_to_bar(params['end'])
            
        return result
    
    def _time_to_bar(self, time_str: str) -> int:
        """Convert time string (HH:MM) to bar number."""
        if ':' not in time_str:
            return int(time_str)  # Already a bar number
            
        hours, minutes = map(int, time_str.split(':'))
        # Convert to minutes from 9:30
        minutes_from_open = (hours - 9) * 60 + minutes - 30
        # Convert to 5-minute bars
        return minutes_from_open // 5


    def _has_parameter_lists(self, strategies: List[Union[Dict, str]]) -> bool:
        """
        Check if any strategy has parameter lists or range() expressions.
        """
        for strategy_def in strategies:
            if isinstance(strategy_def, dict):
                for strategy_type, params in strategy_def.items():
                    if strategy_type == 'threshold':
                        continue
                    if isinstance(params, dict):
                        # Check for any list values or range expressions
                        for key, value in params.items():
                            if key == 'filter':
                                continue
                            if isinstance(value, list):
                                return True
                            if isinstance(value, str) and value.startswith('range('):
                                return True
        return False
    
    def _create_ensemble_config(self, strategies: List[Union[Dict, str]]) -> Dict[str, Any]:
        """
        Create ensemble configuration from strategy list.
        """
        ensemble_strategies = []
        threshold = None
        
        for strategy_def in strategies:
            if isinstance(strategy_def, dict):
                # Check for threshold specification
                if 'threshold' in strategy_def:
                    threshold = strategy_def['threshold']
                    continue
                    
                # Process each strategy
                for strategy_type, params in strategy_def.items():
                    strategy_config = {
                        'type': strategy_type,
                        'name': strategy_type,
                        'param_overrides': params if isinstance(params, dict) else {}
                    }
                    
                    # Handle weight specification
                    if isinstance(params, dict) and 'weight' in params:
                        strategy_config['weight'] = params['weight']
                        # Remove weight from param_overrides
                        strategy_config['param_overrides'] = {k: v for k, v in params.items() if k != 'weight'}
                    elif isinstance(params, dict) and params.get('weight') == '1/n':
                        # Will be processed later
                        strategy_config['weight'] = '1/n'
                        strategy_config['param_overrides'] = {k: v for k, v in params.items() if k != 'weight'}
                    else:
                        # Default weight
                        strategy_config['weight'] = 1.0
                        
                    ensemble_strategies.append(strategy_config)
            else:
                # Simple string strategy
                ensemble_strategies.append({
                    'type': strategy_def,
                    'param_overrides': {},
                    'weight': 1.0
                })
        
        # Handle 1/n weights
        n = len(ensemble_strategies)
        for strategy in ensemble_strategies:
            if strategy.get('weight') == '1/n':
                strategy['weight'] = 1.0 / n
        
        # Create ensemble configuration
        ensemble_config = {
            'strategies': [{
                'type': 'ensemble',
                'name': 'combined_strategies',
                'param_overrides': {
                    'strategies': ensemble_strategies,
                    'combination_method': 'weighted_vote',
                    'threshold': threshold if threshold else 0.0
                }
            }]
        }
        
        return ensemble_config
    
    def _create_ensemble_parameter_space(self, strategies: List[Union[Dict, str]]) -> Dict[str, Any]:
        """
        Create parameter space for ensemble with parameter expansion.
        
        This expands all parameter combinations for each strategy,
        then creates ensembles from the cartesian product.
        """
        import itertools
        
        # First, expand each strategy's parameters
        expanded_by_strategy = []
        constraint = None
        
        for strategy_def in strategies:
            if isinstance(strategy_def, dict):
                # Check for constraint/threshold
                if 'constraints' in strategy_def:
                    constraint = strategy_def['constraints']
                    continue
                elif 'threshold' in strategy_def:
                    constraint = strategy_def['threshold']
                    continue
                    
                for strategy_type, params in strategy_def.items():
                    # Expand this strategy's parameters
                    expanded = self._expand_strategy(strategy_type, params)
                    expanded_by_strategy.append({
                        'strategy_type': strategy_type,
                        'variants': expanded
                    })
        
        # Now create ensemble combinations
        # Get all variant lists
        variant_lists = [s['variants'] for s in expanded_by_strategy]
        
        # Create cartesian product of all variants
        ensemble_strategies = []
        for combo in itertools.product(*variant_lists):
            # Create ensemble from this combination
            ensemble_params = {
                'strategies': list(combo),
                'combination_method': 'weighted_vote',
                'threshold': constraint if constraint else 0.0
            }
            
            ensemble_strategies.append({
                'type': 'ensemble',
                'param_overrides': ensemble_params
            })
        
        logger.info(f"Created {len(ensemble_strategies)} ensemble combinations from parameter expansion")
        
        return {'strategies': ensemble_strategies}


def parse_clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a config with clean syntax and return internal format.
    
    Args:
        config: Config dict with clean syntax
        
    Returns:
        Config dict in internal format
    """
    parser = CleanSyntaxParser()
    return parser.parse_config(config)