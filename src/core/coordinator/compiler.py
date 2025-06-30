"""
Strategy Compiler

Compiles compositional strategy configurations into executable strategies.
Handles the interpretation of the Lisp-like compositional syntax defined
in docs/strategy-composition-design.md.
"""

import logging
import uuid
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from functools import partial

from .protocols import StrategyCompilerProtocol
from ..components.discovery import get_component_registry
from ..features.feature_spec import FeatureSpec
from ...strategy.components.constrained_strategy_wrapper import wrap_strategy_with_filter

logger = logging.getLogger(__name__)


@dataclass
class CompiledStrategy:
    """A compiled strategy ready for execution."""
    id: str
    function: Callable
    features: List[FeatureSpec]
    metadata: Dict[str, Any]


class StrategyCompiler(StrategyCompilerProtocol):
    """
    Compiles compositional strategy configurations into executable functions.
    
    Supports:
    - Atomic strategies (single strategy type)
    - Composite strategies (arrays of strategies)
    - Conditional strategies (with condition expressions)
    - Nested compositions (strategies within strategies)
    - Grid search expansion (parameter space enumeration)
    """
    
    def __init__(self):
        self.registry = get_component_registry()
        self._condition_evaluator = ConditionEvaluator()
    
    def compile_strategies(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compile strategy configuration into executable strategy specs.
        
        For optimization mode with parameter_combinations, expands to many strategies.
        Otherwise returns single compiled strategy.
        
        Also applies execution-level filters like EOD closure if enabled.
        """
        # Store execution config for EOD filter application
        self._execution_config = config.get('execution', {})
        # Check if we have expanded parameter combinations (from parameter_expander)
        if 'parameter_combinations' in config:
            # Grid search mode - compile each combination
            compiled = []
            for combo in config['parameter_combinations']:
                strategy_config = {
                    combo['strategy_type']: {
                        'params': combo['parameters']
                    }
                }
                # Check if this combination has a filter or constraints
                if 'filter' in combo:
                    strategy_config['filter'] = combo['filter']
                if 'filter_params' in combo:
                    strategy_config['filter_params'] = combo['filter_params']
                if 'constraints' in combo:
                    strategy_config['constraints'] = combo['constraints']
                elif 'threshold' in combo:
                    strategy_config['threshold'] = combo['threshold']
                    
                compiled_strategy = self._compile_single(strategy_config)
                
                # Use simple incremental ID
                strategy_type = combo['strategy_type']
                params = combo['parameters']
                strategy_id = f"strategy_{len(compiled)}"
                
                # Generate deterministic hash (same logic as MultiStrategyTracer)
                hash_config = {
                    'type': strategy_type,
                    'parameters': {k: v for k, v in params.items() if not k.startswith('_')}
                }
                # Add filter/constraints if present
                filter_config = combo.get('filter') or combo.get('constraints')
                if filter_config:
                    hash_config['filter'] = filter_config
                    
                config_str = json.dumps(hash_config, sort_keys=True, separators=(',', ':'))
                strategy_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
                
                compiled_strategy['id'] = strategy_id
                # Store the strategy type in metadata for signal tracing
                compiled_strategy['metadata']['strategy_type'] = combo['strategy_type']
                compiled_strategy['metadata']['strategy_name'] = strategy_id  # Use the unique ID
                compiled_strategy['metadata']['strategy_hash'] = strategy_hash  # Add hash for tracing
                # IMPORTANT: Store the actual parameters for metadata.json
                compiled_strategy['metadata']['parameters'] = combo['parameters']
                compiled_strategy['metadata']['filter'] = combo.get('filter')
                compiled_strategy['metadata']['filter_params'] = combo.get('filter_params')
                
                # Also store parameters directly on the function for easier access
                compiled_strategy['function']._compiled_params = combo['parameters']
                
                compiled.append(compiled_strategy)
            return compiled
        else:
            # Single strategy mode
            strategy_config = config.get('strategy', config)
            compiled = self._compile_single(strategy_config)
            
            # If the strategy is a composite, update its ID to be more descriptive
            if compiled['metadata'].get('composition_type') == 'composite' and 'composite_strategies' in compiled['metadata']:
                # Create a descriptive ID from the sub-strategies
                sub_types = [s['type'] for s in compiled['metadata']['composite_strategies']]
                compiled['id'] = f"ensemble_{'_'.join(sub_types[:3])}"  # Limit to first 3 for brevity
                compiled['metadata']['strategy_type'] = 'ensemble'
                compiled['metadata']['strategy_name'] = compiled['id']
                
                # For single-strategy ensembles, also expose the parameters at the top level
                if len(compiled['metadata']['composite_strategies']) == 1:
                    first_strategy = compiled['metadata']['composite_strategies'][0]
                    compiled['metadata']['parameters'] = first_strategy.get('params', {})
                    logger.info(f"[COMPILER] Single-strategy ensemble, exposing parameters at top level: {compiled['metadata']['parameters']}")
                
            return [compiled]
    
    def extract_features(self, config: Dict[str, Any]) -> List[FeatureSpec]:
        """Recursively extract all features needed by the strategy configuration."""
        # Handle parameter combinations case
        if 'parameter_combinations' in config:
            # Extract features from ALL combinations to get complete set
            all_features = {}
            for combo in config['parameter_combinations']:
                strategy_config = {
                    combo['strategy_type']: {
                        'params': combo['parameters']
                    }
                }
                # Include filter in strategy config for feature extraction
                if 'filter' in combo:
                    strategy_config['filter'] = combo['filter']
                if 'filter_params' in combo:
                    strategy_config['filter_params'] = combo['filter_params']
                    
                features = self._extract_features_recursive(strategy_config)
                # Add to set of unique features
                for feature in features:
                    all_features[feature.canonical_name] = feature
            return list(all_features.values())
        else:
            strategy_config = config.get('strategy', config)
            return self._extract_features_recursive(strategy_config)
    
    def _compile_single(self, config: Any) -> Dict[str, Any]:
        """Compile a single strategy configuration."""
        # Generate unique ID
        strategy_id = str(uuid.uuid4())[:8]
        
        # Compile the strategy function
        compiled_func = self._compile_recursive(config)
        
        # Check if there's a filter, threshold, or constraints in the config and wrap the function
        if isinstance(config, dict):
            filter_expr = None
            
            # Support 'constraints' (new), 'threshold' (transitional), and 'filter' (deprecated) fields
            if 'constraints' in config:
                filter_expr = config['constraints']
                logger.info(f"[COMPILER] Using constraints: {filter_expr}")
            elif 'threshold' in config:
                filter_expr = config['threshold']
                logger.info(f"[COMPILER] Using threshold: {filter_expr}")
            elif 'filter' in config:
                filter_expr = config['filter']
                logger.info(f"[COMPILER] Using deprecated filter: {filter_expr}")
            
            # Apply EOD threshold if execution.close_eod is enabled
            if hasattr(self, '_execution_config') and self._execution_config.get('close_eod'):
                eod_threshold = "(time < 1550)"  # Close all positions by 3:50 PM
                if filter_expr:
                    # Combine with existing threshold
                    filter_expr = f"({filter_expr}) and {eod_threshold}"
                    logger.info(f"[COMPILER] Added EOD threshold to existing threshold: {filter_expr}")
                else:
                    filter_expr = eod_threshold
                    logger.info(f"[COMPILER] Applied EOD threshold: {filter_expr}")
            
            if filter_expr:
                # Use 'constraints' key for the filter config
                filter_config = {'constraints': filter_expr}
                if 'filter_params' in config:
                    filter_config['filter_params'] = config['filter_params']
                compiled_func = wrap_strategy_with_filter(compiled_func, filter_config)
                logger.info(f"Applied constraints to strategy: {filter_expr}")
        
        # Extract features
        features = self._extract_features_recursive(config)
        
        # Build metadata
        metadata = {
            'composition_type': self._get_composition_type(config),
            'config': config,
            'feature_specs': features  # Include feature specs for discovery
        }
        
        # If this is a composite strategy, track sub-strategies
        if isinstance(config, list):
            metadata['composite_strategies'] = []
            for sub_config in config:
                if isinstance(sub_config, dict):
                    # Extract strategy info from each sub-strategy
                    for strategy_type, params in sub_config.items():
                        if strategy_type not in ['weight', 'threshold', 'condition']:
                            sub_strategy_info = {
                                'type': strategy_type,
                                'params': params if isinstance(params, dict) else {}
                            }
                            metadata['composite_strategies'].append(sub_strategy_info)
                            logger.info(f"[COMPILER] Added sub-strategy to composite: {strategy_type} with params: {sub_strategy_info['params']}")
        
        # Try to get original strategy metadata if this is an atomic strategy
        if isinstance(config, dict) and len(config) == 1:
            strategy_type = list(config.keys())[0]
            strategy_info = self.registry.get_component(strategy_type)
            # Store the strategy type in metadata
            metadata['strategy_type'] = strategy_type
            metadata['strategy_name'] = strategy_type
            if strategy_info and strategy_info.metadata:
                # Copy relevant metadata from original strategy
                original_metadata = strategy_info.metadata
                if 'feature_discovery' in original_metadata:
                    metadata['feature_discovery'] = original_metadata['feature_discovery']
                    logger.debug(f"Copied feature_discovery for {strategy_type}")
                if 'required_features' in original_metadata:
                    metadata['required_features'] = original_metadata['required_features']
                    logger.debug(f"Copied required_features for {strategy_type}")
            else:
                logger.warning(f"No metadata found for strategy type: {strategy_type}")
        
        # Attach metadata to the function itself for event system
        compiled_func._strategy_metadata = metadata
        
        return {
            'id': strategy_id,
            'function': compiled_func,
            'features': features,
            'metadata': metadata
        }
    
    def _compile_recursive(self, config: Any) -> Callable:
        """Recursively compile strategy configuration into executable function."""
        
        # Array = composite strategy
        if isinstance(config, list):
            return self._compile_composite(config)
        
        # Dict with various patterns
        elif isinstance(config, dict):
            # Multi-state regime with cases
            if 'regime' in config and 'cases' in config:
                return self._compile_regime_cases(config)
            
            # Multi-state regime with nested classifiers
            elif 'regimes' in config and 'cases' in config:
                return self._compile_multi_regime(config)
            
            # Multiple conditions (for same expression, different values)
            elif 'conditions' in config and isinstance(config['conditions'], list):
                return self._compile_multi_condition(config)
            
            # Conditional with if_true/if_false
            elif 'if_true' in config and 'if_false' in config:
                return self._compile_conditional_branch(config)
            
            # Conditional in list (condition + strategy)
            elif 'condition' in config:
                return self._compile_conditional(config)
            
            # Container with nested strategies
            elif 'strategy' in config:
                nested_func = self._compile_recursive(config['strategy'])
                # Pass through any container-level settings
                return partial(self._apply_container_settings, 
                             nested_func=nested_func,
                             settings=config)
            
            # Dict with 'strategies' array (alternative composite syntax)
            elif 'strategies' in config:
                return self._compile_composite(config['strategies'], config)
            
            # Atomic strategy (has strategy type as key)
            else:
                return self._compile_atomic(config)
        
        else:
            raise ValueError(f"Unknown strategy configuration type: {type(config)}")
    
    def _compile_atomic(self, config: Dict[str, Any]) -> Callable:
        """Compile atomic strategy configuration."""
        # Refresh registry to ensure we have latest components
        self.registry = get_component_registry()
        
        logger.debug(f"_compile_atomic called with config keys: {list(config.keys())}")
        logger.debug(f"Full config: {config}")
        
        # Check if there's a constraint/threshold at this level
        # Support both 'constraints' (new) and 'threshold' (deprecated) 
        constraint_expr = config.get('constraints') or config.get('threshold')
        
        # Find the strategy type key (first key that's in registry)
        strategy_type = None
        strategy_config = None
        
        # Skip known non-strategy keys
        skip_keys = {'weight', 'condition', 'combination', 'weight_threshold', 'if_true', 'if_false', 'strategy', 'strategies', 'threshold', 'constraints', 'type', 'param_overrides'}
        
        # Check if this is the parameter_space format with 'type' and 'param_overrides'
        if 'type' in config and 'param_overrides' in config:
            strategy_type = config['type']
            strategy_config = config['param_overrides']
        else:
            # Look for strategy type as a key
            for key, value in config.items():
                if key in skip_keys:
                    continue
                    
                # Check if this key is a registered strategy
                component_info = self.registry.get_component(key)
                if component_info and component_info.component_type == 'strategy':
                    strategy_type = key
                    strategy_config = value if isinstance(value, dict) else {}
                    break
        
        if not strategy_type:
            # Log available strategies for debugging
            available_strategies = [name for name, info in self.registry._components.items() 
                                  if info.component_type == 'strategy']
            logger.error(f"Registry has {len(self.registry._components)} total components")
            logger.error(f"Available strategies in registry: {available_strategies[:10]}...")
            raise ValueError(f"No valid strategy type found in config: {list(config.keys())}")
        
        # Get strategy function from registry
        strategy_info = self.registry.get_component(strategy_type)
        if not strategy_info:
            raise ValueError(f"Strategy '{strategy_type}' not found in registry")
        
        strategy_func = strategy_info.factory
        if not strategy_func:
            raise ValueError(f"Strategy '{strategy_type}' has no factory function")
        
        # Handle different parameter formats
        if 'params' in strategy_config:
            # Old format: {strategy_name: {params: {...}}}
            params = strategy_config['params']
        else:
            # New format: {strategy_name: {...params...}}
            # Extract all non-metadata keys as params
            params = {k: v for k, v in strategy_config.items() 
                     if k not in {'weight', 'condition', 'metadata'}}
        
        # Validate parameters against parameter_space if defined
        if hasattr(strategy_info, 'metadata') and 'parameter_space' in strategy_info.metadata:
            param_space = strategy_info.metadata['parameter_space']
            if param_space:
                # Check for unknown parameters
                valid_params = set(param_space.keys())
                provided_params = set(params.keys())
                # Allow special parameters that start with underscore (like _risk)
                unknown_params = {p for p in provided_params - valid_params if not p.startswith('_')}
                
                if unknown_params:
                    raise ValueError(
                        f"Strategy '{strategy_type}' received unknown parameters: {unknown_params}. "
                        f"Valid parameters are: {sorted(valid_params)}"
                    )
                
                # Check for required parameters (those without defaults)
                required_params = {
                    name for name, spec in param_space.items()
                    if 'default' not in spec
                }
                missing_params = required_params - provided_params
                
                if missing_params:
                    raise ValueError(
                        f"Strategy '{strategy_type}' missing required parameters: {missing_params}. "
                        f"Got: {list(provided_params)}"
                    )
        
        # Return wrapped function that applies parameters
        def atomic_executor(features, bar, runtime_params):
            # Use the compiled params, not runtime params (which may be empty)
            # This ensures we use the params from the config file
            actual_params = params.copy()
            
            # Log what we're actually using
            logger.debug(f"Atomic executor for {strategy_type}: using params={actual_params}")
            
            # FAIL if required params are missing instead of using defaults
            if strategy_type == 'bollinger_bands':
                if 'period' not in actual_params or 'std_dev' not in actual_params:
                    raise ValueError(f"Bollinger Bands missing required parameters! Got: {actual_params}")
                # Check we have the right features
                period = actual_params['period']
                std_dev = actual_params['std_dev']
                upper = features.get(f'bollinger_bands_{period}_{std_dev}_upper')
                lower = features.get(f'bollinger_bands_{period}_{std_dev}_lower')
                if upper is None or lower is None:
                    logger.error(f"Missing Bollinger features for period={period}, std_dev={std_dev}")
                    
            elif strategy_type == 'keltner_bands' or strategy_type == 'keltner_breakout':
                if 'period' not in actual_params or 'multiplier' not in actual_params:
                    raise ValueError(f"Keltner {strategy_type} missing required parameters! Got: {actual_params}")
                # Check we have the right features
                period = actual_params['period']
                multiplier = actual_params['multiplier']
                # Keltner uses alphabetical sorting: multiplier before period
                upper = features.get(f'keltner_channel_{multiplier}_{period}_upper')
                lower = features.get(f'keltner_channel_{multiplier}_{period}_lower')
                if upper is None or lower is None:
                    logger.error(f"Missing Keltner features for period={period}, multiplier={multiplier}")
                    
            elif strategy_type == 'donchian_bands' or strategy_type == 'donchian_breakout':
                if 'period' not in actual_params:
                    raise ValueError(f"Donchian {strategy_type} missing required parameter 'period'! Got: {actual_params}")
                    
            elif strategy_type == 'rsi_oversold' or strategy_type == 'rsi_overbought':
                if 'period' not in actual_params:
                    raise ValueError(f"RSI {strategy_type} missing required parameter 'period'! Got: {actual_params}")
                    
            elif strategy_type == 'ma_crossover':
                if 'fast_period' not in actual_params or 'slow_period' not in actual_params:
                    raise ValueError(f"MA Crossover missing required parameters! Got: {actual_params}")
                    
            elif strategy_type == 'macd':
                if 'fast_period' not in actual_params or 'slow_period' not in actual_params or 'signal_period' not in actual_params:
                    raise ValueError(f"MACD missing required parameters! Got: {actual_params}")
                    
            result = strategy_func(features, bar, actual_params)
            if result is None:
                logger.debug(f"Strategy {strategy_type} returned None")
            return result
        
        # Preserve the original strategy's metadata on the wrapper
        if hasattr(strategy_func, '_component_info'):
            atomic_executor._component_info = strategy_func._component_info
        
        # Wrap with constraint filter if constraint is specified
        if constraint_expr:
            logger.info(f"[COMPILER] Wrapping {strategy_type} with constraint: {constraint_expr}")
            from src.strategy.components.constrained_strategy_wrapper import ConstrainedStrategyWrapper
            from src.strategy.components.config_filter import ConfigSignalFilter
            
            # Handle multiple constraints - combine with AND by default
            if isinstance(constraint_expr, list):
                combined_expr = " and ".join(f"({expr})" for expr in constraint_expr)
                logger.info(f"[COMPILER] Combined {len(constraint_expr)} constraints with AND: {combined_expr}")
                constraint_expr = combined_expr
            
            # Wrap the strategy with the constraint expression
            wrapped_strategy = ConstrainedStrategyWrapper(
                atomic_executor,
                filter_expr=constraint_expr
            )
            
            # Create a new executor that uses the wrapped strategy
            def constraint_executor(features, bar, runtime_params):
                return wrapped_strategy(features, bar, runtime_params)
            
            # Preserve metadata
            if hasattr(atomic_executor, '_component_info'):
                constraint_executor._component_info = atomic_executor._component_info
            
            return constraint_executor
        
        return atomic_executor
    
    def _compile_composite(self, strategies: List[Any], 
                          container_config: Optional[Dict[str, Any]] = None) -> Callable:
        """Compile composite strategy from list of strategies."""
        # Extract global constraint/threshold if present
        global_constraint = None
        actual_strategies = []
        
        for item in strategies:
            if isinstance(item, dict) and len(item) == 1:
                if 'constraints' in item:
                    # This is a global constraints object
                    global_constraint = item['constraints']
                    logger.info(f"[COMPILER] Found global constraints: {global_constraint}")
                elif 'threshold' in item:
                    # Deprecated but still supported
                    global_constraint = item['threshold']
                    logger.info(f"[COMPILER] Found global threshold (deprecated): {global_constraint}")
                else:
                    # This is an actual strategy
                    actual_strategies.append(item)
            else:
                # This is an actual strategy
                actual_strategies.append(item)
        
        # Compile each sub-strategy
        compiled_strategies = []
        weights = []
        
        for idx, strategy_config in enumerate(actual_strategies):
            # Apply global constraint if strategy doesn't have its own
            if global_constraint and isinstance(strategy_config, dict):
                has_own_constraint = False
                # Check if strategy has its own constraint/threshold
                if 'constraints' in strategy_config or 'threshold' in strategy_config:
                    has_own_constraint = True
                else:
                    # Check inside strategy definition
                    for key, value in strategy_config.items():
                        if key not in ['weight', 'condition'] and isinstance(value, dict):
                            if 'constraints' in value or 'threshold' in value:
                                has_own_constraint = True
                                break
                
                if not has_own_constraint:
                    # Apply global constraint
                    strategy_config = strategy_config.copy()
                    strategy_config['constraints'] = global_constraint
                    logger.info(f"[COMPILER] Applied global constraint to strategy {idx}")
            
            compiled_func = self._compile_recursive(strategy_config)
            
            # Extract weight if present
            weight = 1.0
            if isinstance(strategy_config, dict):
                # Check various places weight might be
                if 'weight' in strategy_config:
                    weight = strategy_config['weight']
                else:
                    # Look inside the strategy definition
                    for key, value in strategy_config.items():
                        if isinstance(value, dict) and 'weight' in value:
                            weight = value['weight']
                            break
            
            compiled_strategies.append(compiled_func)
            weights.append(weight)
        
        # Get combination settings
        combination_method = 'weighted_vote'
        weight_threshold = 0.5
        
        if container_config:
            combination_method = container_config.get('combination', combination_method)
            weight_threshold = container_config.get('weight_threshold', weight_threshold)
        
        # Return composite executor
        def composite_executor(features, bar, params):
            return self._execute_composite(
                compiled_strategies, weights, features, bar, params,
                combination_method, weight_threshold
            )
        
        # Store sub-strategy info for potential attribution tracking
        composite_executor._sub_strategies = []
        for i, strategy_config in enumerate(strategies):
            if isinstance(strategy_config, dict):
                for strategy_type, strategy_params in strategy_config.items():
                    if strategy_type not in ['weight', 'threshold', 'condition']:
                        composite_executor._sub_strategies.append({
                            'type': strategy_type,
                            'weight': weights[i] if i < len(weights) else 1.0,
                            'params': strategy_params if isinstance(strategy_params, dict) else {}
                        })
        
        return composite_executor
    
    def _compile_conditional(self, config: Dict[str, Any]) -> Callable:
        """Compile conditional strategy."""
        condition_expr = config['condition']
        
        # Compile the strategy part
        strategy_config = {k: v for k, v in config.items() if k != 'condition'}
        strategy_func = self._compile_recursive(strategy_config)
        
        # Return conditional executor
        def conditional_executor(features, bar, params):
            # Evaluate condition
            context = self._build_condition_context(features, bar, params)
            if self._condition_evaluator.evaluate(condition_expr, context):
                return strategy_func(features, bar, params)
            else:
                return None  # Strategy not active
        
        return conditional_executor
    
    def _compile_conditional_branch(self, config: Dict[str, Any]) -> Callable:
        """Compile if/then/else conditional strategy."""
        condition_expr = config['condition']
        if_true_func = self._compile_recursive(config['if_true'])
        if_false_func = self._compile_recursive(config['if_false'])
        
        # Return branching executor
        def branch_executor(features, bar, params):
            context = self._build_condition_context(features, bar, params)
            if self._condition_evaluator.evaluate(condition_expr, context):
                return if_true_func(features, bar, params)
            else:
                return if_false_func(features, bar, params)
        
        return branch_executor
    
    def _compile_regime_cases(self, config: Dict[str, Any]) -> Callable:
        """Compile regime-based strategy with cases for each state."""
        regime_expr = config['regime']
        cases = config['cases']
        
        # Compile strategy for each case
        case_strategies = {}
        for state, strategy_config in cases.items():
            case_strategies[state] = self._compile_recursive(strategy_config)
        
        # Return regime executor
        def regime_executor(features, bar, params):
            # Evaluate regime classifier
            context = self._build_condition_context(features, bar, params)
            regime_state = self._condition_evaluator.evaluate_classifier(regime_expr, context)
            
            # Execute strategy for current regime
            if regime_state in case_strategies:
                return case_strategies[regime_state](features, bar, params)
            else:
                logger.warning(f"No strategy defined for regime state: {regime_state}")
                return None
        
        return regime_executor
    
    def _compile_multi_regime(self, config: Dict[str, Any]) -> Callable:
        """Compile multi-regime strategy with nested classifier states."""
        regimes = config['regimes']  # Dict of classifier names to expressions
        cases = config['cases']  # Nested dict of states
        
        # This handles nested regime matching like:
        # volatility: {low: {trend: {strong: strategy1, weak: strategy2}}}
        
        def multi_regime_executor(features, bar, params):
            context = self._build_condition_context(features, bar, params)
            
            # Evaluate all regime classifiers
            regime_states = {}
            for name, expr in regimes.items():
                regime_states[name] = self._condition_evaluator.evaluate_classifier(expr, context)
            
            # Navigate nested cases to find matching strategy
            current_cases = cases
            for regime_name, state in regime_states.items():
                if regime_name in current_cases:
                    current_cases = current_cases[regime_name]
                    if state in current_cases:
                        current_cases = current_cases[state]
                    else:
                        logger.warning(f"No case for {regime_name}={state}")
                        return None
                else:
                    break
            
            # If we've navigated to a strategy config, compile and execute it
            if isinstance(current_cases, dict) and not any(k in regime_states for k in current_cases):
                # This is a strategy config, not more regime cases
                strategy_func = self._compile_recursive(current_cases)
                return strategy_func(features, bar, params)
            else:
                logger.warning(f"Incomplete regime case matching")
                return None
        
        return multi_regime_executor
    
    def _compile_multi_condition(self, config: Dict[str, Any]) -> Callable:
        """Compile strategy with multiple conditions (typically for different weights)."""
        conditions = config['conditions']
        strategy_config = {k: v for k, v in config.items() if k != 'conditions'}
        
        # Compile the base strategy
        base_strategy = self._compile_recursive(strategy_config)
        
        def multi_condition_executor(features, bar, params):
            context = self._build_condition_context(features, bar, params)
            
            # Find first matching condition
            active_weight = None
            for cond_spec in conditions:
                if self._condition_evaluator.evaluate(cond_spec['condition'], context):
                    active_weight = cond_spec.get('weight', 1.0)
                    break
            
            if active_weight is None:
                return None  # No conditions matched
            
            # Execute base strategy
            result = base_strategy(features, bar, params)
            
            # Apply weight from matching condition
            if result and 'metadata' not in result:
                result['metadata'] = {}
            if result:
                result['metadata']['condition_weight'] = active_weight
            
            return result
        
        return multi_condition_executor
    
    def _execute_composite(self, strategies: List[Callable], weights: List[float],
                          features: Dict[str, Any], bar: Dict[str, Any], 
                          params: Dict[str, Any], combination_method: str,
                          weight_threshold: float) -> Optional[Dict[str, Any]]:
        """Execute composite strategy and combine signals."""
        # Execute all sub-strategies
        signals = []
        active_weights = []
        
        for strategy_func, weight in zip(strategies, weights):
            try:
                signal = strategy_func(features, bar, params)
                if signal and signal.get('signal_value') is not None:
                    signals.append(signal)
                    active_weights.append(weight)
            except Exception as e:
                logger.error(f"Error executing sub-strategy: {e}")
                continue
        
        if not signals:
            return None
        
        # Normalize weights
        total_weight = sum(active_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in active_weights]
        else:
            normalized_weights = [1.0 / len(signals) for _ in signals]
        
        # Combine signals based on method
        combined_value, confidence = self._combine_signals(
            signals, normalized_weights, combination_method
        )
        
        # Apply weight threshold
        if confidence < weight_threshold:
            combined_value = 0
        
        # Build composite signal
        return {
            'signal_value': combined_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'composite',
            'symbol_timeframe': f"{bar.get('symbol', 'UNKNOWN')}_{bar.get('timeframe', '5m')}",
            'metadata': {
                'num_signals': len(signals),
                'combination_method': combination_method,
                'confidence': confidence,
                'sub_signals': [s.get('signal_value', 0) for s in signals]
            }
        }
    
    def _combine_signals(self, signals: List[Dict[str, Any]], 
                        weights: List[float], method: str) -> Tuple[int, float]:
        """Combine signals using specified method."""
        signal_values = [s.get('signal_value', 0) for s in signals]
        
        if method == 'weighted_vote':
            weighted_sum = sum(v * w for v, w in zip(signal_values, weights))
            if weighted_sum > 0:
                return (1, weighted_sum)
            elif weighted_sum < 0:
                return (-1, abs(weighted_sum))
            else:
                return (0, 0.0)
        
        elif method == 'majority':
            bullish = sum(1 for v in signal_values if v > 0)
            bearish = sum(1 for v in signal_values if v < 0)
            total = len(signal_values)
            
            if bullish > total / 2:
                return (1, bullish / total)
            elif bearish > total / 2:
                return (-1, bearish / total)
            else:
                return (0, 0.0)
        
        elif method == 'unanimous':
            if all(v > 0 for v in signal_values):
                return (1, 1.0)
            elif all(v < 0 for v in signal_values):
                return (-1, 1.0)
            else:
                return (0, 0.0)
        
        elif method == 'average':
            avg = sum(signal_values) / len(signal_values)
            if avg > 0:
                return (1, avg)
            elif avg < 0:
                return (-1, abs(avg))
            else:
                return (0, 0.0)
        
        else:
            logger.warning(f"Unknown combination method: {method}")
            return (0, 0.0)
    
    def _extract_features_recursive(self, config: Any) -> List[FeatureSpec]:
        """Recursively extract features from strategy configuration."""
        features = []
        
        if isinstance(config, list):
            # Composite - extract from all strategies
            for strategy_config in config:
                features.extend(self._extract_features_recursive(strategy_config))
        
        elif isinstance(config, dict):
            # Check various patterns
            if 'if_true' in config and 'if_false' in config:
                # Conditional branch - need features from both branches
                features.extend(self._extract_features_recursive(config['if_true']))
                features.extend(self._extract_features_recursive(config['if_false']))
            
            elif 'condition' in config:
                # Extract features from condition expression
                condition_features = self._extract_condition_features(config['condition'])
                if condition_features:
                    features.extend(condition_features)
                # And from the strategy part
                strategy_config = {k: v for k, v in config.items() if k != 'condition'}
                recursive_features = self._extract_features_recursive(strategy_config)
                if recursive_features:
                    features.extend(recursive_features)
            
            elif 'strategy' in config:
                # Nested strategies
                features.extend(self._extract_features_recursive(config['strategy']))
            
            elif 'strategies' in config:
                # Alternative composite syntax
                features.extend(self._extract_features_recursive(config['strategies']))
            
            else:
                # Atomic strategy
                atomic_features = self._extract_atomic_features(config)
                if atomic_features:
                    features.extend(atomic_features)
                
                # NEW: Extract features from filter, threshold, or constraints if present
                if 'filter' in config or 'threshold' in config or 'constraints' in config:
                    from .feature_discovery import FeatureDiscovery
                    discovery = FeatureDiscovery()
                    
                    # Get the constraint/filter expressions
                    constraint_exprs = []
                    
                    if 'constraints' in config:
                        # Handle both single constraint and array of constraints
                        constraints = config['constraints']
                        if isinstance(constraints, str):
                            constraint_exprs.append(constraints)
                        elif isinstance(constraints, list):
                            constraint_exprs.extend(constraints)
                    elif 'threshold' in config:
                        # Deprecated but still supported
                        threshold = config['threshold']
                        if isinstance(threshold, str):
                            constraint_exprs.append(threshold)
                        elif isinstance(threshold, list):
                            constraint_exprs.extend(threshold)
                    elif 'filter' in config:
                        # Legacy filter support
                        constraint_exprs.append(config['filter'])
                    
                    # Discover features from all constraint expressions
                    for expr in constraint_exprs:
                        if expr:
                            constraint_features = discovery._discover_features_from_filter(expr)
                            if constraint_features:
                                logger.info(f"Discovered {len(constraint_features)} features from constraint: {expr}")
                                features.extend(constraint_features)
        
        # Deduplicate features
        unique_features = {}
        for feature in features:
            key = feature.canonical_name
            if key not in unique_features:
                unique_features[key] = feature
        
        return list(unique_features.values())
    
    def _extract_atomic_features(self, config: Dict[str, Any]) -> List[FeatureSpec]:
        """Extract features from atomic strategy configuration."""
        # Find strategy type
        strategy_type = None
        strategy_config = None
        
        for key, value in config.items():
            # Check if this key is a registered strategy
            component_info = self.registry.get_component(key)
            if component_info and component_info.component_type == 'strategy':
                strategy_type = key
                strategy_config = value if isinstance(value, dict) else {}
                break
        
        if not strategy_type:
            return []
        
        # Get strategy info
        strategy_info = self.registry.get_component(strategy_type)
        if not strategy_info:
            return []
        
        # Use feature discovery if available
        feature_discovery = strategy_info.metadata.get('feature_discovery')
        if feature_discovery:
            # Handle different parameter formats
            if 'params' in strategy_config:
                params = strategy_config['params']
            else:
                # Extract all non-metadata keys as params
                params = {k: v for k, v in strategy_config.items() 
                         if k not in {'weight', 'condition', 'metadata'}}
            return feature_discovery(params)
        
        # Fall back to static features
        return strategy_info.metadata.get('required_features', [])
    
    def _extract_condition_features(self, condition_expr: str) -> List[FeatureSpec]:
        """Extract features referenced in condition expression."""
        # This is a simplified extraction - in practice would need proper parsing
        features = []
        
        # Look for indicator function calls like "rsi(14)", "sma(20)", etc.
        import re
        pattern = r'(\w+)\(([^)]+)\)'
        matches = re.findall(pattern, condition_expr)
        
        for func_name, args in matches:
            # Check if it's a known indicator
            if func_name in ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 
                           'adx', 'volatility_percentile', 'trend_strength']:
                # Parse arguments (simplified - assumes single numeric arg)
                try:
                    if ',' in args:
                        # Multiple args
                        arg_list = [a.strip() for a in args.split(',')]
                        params = {}
                        for i, arg in enumerate(arg_list):
                            if arg.isdigit():
                                params[f'param_{i}'] = int(arg)
                    else:
                        # Single arg
                        period = int(args.strip())
                        params = {'period': period}
                    
                    features.append(FeatureSpec(func_name, params))
                except:
                    logger.warning(f"Could not parse condition feature: {func_name}({args})")
        
        return features
    
    def _build_condition_context(self, features: Dict[str, Any], 
                               bar: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for condition evaluation."""
        return {
            'features': features,
            'bar': bar,
            'params': params,
            'price': bar.get('close', 0),
            'volume': bar.get('volume', 0),
            'timestamp': bar.get('timestamp')
        }
    
    def _get_composition_type(self, config: Any) -> str:
        """Determine the composition type of a strategy configuration."""
        if isinstance(config, list):
            return 'composite'
        elif isinstance(config, dict):
            if 'regime' in config and 'cases' in config:
                return 'regime_based'
            elif 'if_true' in config and 'if_false' in config:
                return 'conditional_branch'
            elif 'condition' in config:
                return 'conditional'
            elif 'strategy' in config or 'strategies' in config:
                return 'container'
            else:
                return 'atomic'
        else:
            return 'unknown'
    
    def _apply_container_settings(self, features: Dict[str, Any], bar: Dict[str, Any],
                                 params: Dict[str, Any], nested_func: Callable,
                                 settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply container-level settings to nested strategy result."""
        result = nested_func(features, bar, params)
        
        if result and settings:
            # Could apply transformations based on container settings
            # For now, just pass through
            pass
        
        return result
    
    def _get_composition_type(self, config: Any) -> str:
        """Determine the type of composition."""
        if isinstance(config, list):
            return 'composite'
        elif isinstance(config, dict):
            if 'condition' in config:
                return 'conditional'
            elif 'if_true' in config:
                return 'conditional_branch'
            elif 'strategy' in config or 'strategies' in config:
                return 'nested'
            else:
                return 'atomic'
        return 'unknown'


class ConditionEvaluator:
    """Evaluates condition expressions safely."""
    
    def evaluate(self, expression: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition expression in given context.
        
        Supports:
        - Indicator functions: sma(20), rsi(14), etc.
        - Comparison operators: >, <, ==, !=, >=, <=
        - Logical operators: and, or, not
        - Market state functions: market_hours(), time_until_close()
        """
        try:
            # Build safe evaluation context
            safe_context = self._build_safe_context(context)
            
            # Evaluate expression with restricted builtins
            return eval(expression, {"__builtins__": {}}, safe_context)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{expression}': {e}")
            return False
    
    def evaluate_classifier(self, expression: str, context: Dict[str, Any]) -> str:
        """
        Evaluate classifier expression that returns a state string.
        
        Examples:
        - market_regime_classifier() -> 'trending_up', 'trending_down', 'ranging'
        - volatility_regime(20) -> 'low', 'medium', 'high'
        """
        try:
            # Build safe evaluation context with classifier functions
            safe_context = self._build_safe_context(context)
            
            # Add classifier functions that return states
            safe_context.update({
                'market_regime_classifier': lambda: self._get_market_regime(context),
                'volatility_regime_classifier': lambda: self._get_volatility_regime(context),
                'volatility_regime': lambda period: self._get_volatility_regime(context, period),
                'trend_strength_classifier': lambda: self._get_trend_regime(context),
                'volatility_momentum_classifier': lambda: self._get_volatility_momentum_regime(context),
            })
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_context)
            
            # Ensure we got a string result
            if not isinstance(result, str):
                logger.warning(f"Classifier '{expression}' returned non-string: {result}")
                return 'unknown'
                
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating classifier '{expression}': {e}")
            return 'unknown'
    
    def _build_safe_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build safe evaluation context with common functions."""
        return {
            # Data access
            'price': context.get('price', 0),
            'volume': context.get('volume', 0),
            'features': context.get('features', {}),
            
            # Indicator functions (simplified - would need actual implementation)
            'sma': lambda period: context['features'].get(f'sma_{period}', 0),
            'ema': lambda period: context['features'].get(f'ema_{period}', 0),
            'rsi': lambda period: context['features'].get(f'rsi_{period}', 50),
            'volatility_percentile': lambda period: context['features'].get(f'volatility_percentile_{period}', 50),
            'trend_strength': lambda period: context['features'].get(f'trend_strength_{period}', 0),
            'adx': lambda period: context['features'].get(f'adx_{period}', 0),
            
            # Market state functions
            'market_hours': lambda: 'regular',  # Simplified
            'time_until_close': lambda: 390,  # Minutes
        }
    
    def _get_market_regime(self, context: Dict[str, Any]) -> str:
        """Determine market regime from features."""
        # Simplified implementation - would use actual classifier
        features = context.get('features', {})
        
        # Check trend indicators
        sma_50 = features.get('sma_50', 0)
        sma_200 = features.get('sma_200', 0)
        adx_14 = features.get('adx_14', 0)
        
        if adx_14 > 25:
            if sma_50 > sma_200:
                return 'trending_up'
            else:
                return 'trending_down'
        else:
            return 'ranging'
    
    def _get_volatility_regime(self, context: Dict[str, Any], period: int = 20) -> str:
        """Determine volatility regime from features."""
        features = context.get('features', {})
        vol_percentile = features.get(f'volatility_percentile_{period}', 50)
        
        if vol_percentile < 30:
            return 'low'
        elif vol_percentile < 70:
            return 'medium'
        else:
            return 'high'
    
    def _get_trend_regime(self, context: Dict[str, Any]) -> str:
        """Determine trend strength regime."""
        features = context.get('features', {})
        trend_strength = features.get('trend_strength_50', 0)
        
        if abs(trend_strength) < 0.3:
            return 'weak'
        elif abs(trend_strength) < 0.7:
            return 'moderate'
        else:
            return 'strong'
    
    def _get_volatility_momentum_regime(self, context: Dict[str, Any]) -> str:
        """Determine volatility-momentum regime using the classifier logic."""
        features = context.get('features', {})
        
        # Get features (using default periods)
        atr = features.get('atr_14', 0)
        rsi = features.get('rsi_14', 50)
        sma = features.get('sma_20', 0)
        price = context.get('price', 0)
        
        if not all([atr, sma, price > 0]):
            return 'neutral'
        
        # Calculate volatility level (relative to price)
        vol_pct = (atr / price) * 100
        is_high_vol = vol_pct > 1.0  # 1% threshold
        
        # Determine momentum state
        if rsi > 60:
            momentum_state = 'bullish'
        elif rsi < 40:
            momentum_state = 'bearish'
        else:
            momentum_state = 'neutral'
        
        # Combine volatility and momentum
        if is_high_vol:
            if momentum_state == 'bullish':
                return 'high_vol_bullish'
            elif momentum_state == 'bearish':
                return 'high_vol_bearish'
            else:
                return 'neutral'
        else:
            if momentum_state == 'bullish':
                return 'low_vol_bullish'
            elif momentum_state == 'bearish':
                return 'low_vol_bearish'
            else:
                return 'neutral'