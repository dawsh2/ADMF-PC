"""
CLI parameter parser for strategy and classifier specifications.

Converts CLI string specifications into configuration dictionaries
that can be processed by the topology builder.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_strategy_specs(strategy_specs: List[str]) -> List[Dict[str, Any]]:
    """
    Parse strategy specifications from CLI strings.
    
    Format: "type:param1=val1,val2;param2=val3"
    Example: "momentum:lookback=10,20,30;threshold=0.01,0.02"
    
    Args:
        strategy_specs: List of strategy specification strings
        
    Returns:
        List of strategy configuration dictionaries
    """
    strategies = []
    
    for spec in strategy_specs:
        try:
            strategy_config = _parse_component_spec(spec, 'strategy')
            if strategy_config:
                strategies.append(strategy_config)
        except Exception as e:
            logger.error(f"Failed to parse strategy spec '{spec}': {e}")
            continue
    
    logger.info(f"Parsed {len(strategies)} strategy specifications")
    return strategies


def parse_classifier_specs(classifier_specs: List[str]) -> List[Dict[str, Any]]:
    """
    Parse classifier specifications from CLI strings.
    
    Format: "type:param1=val1,val2;param2=val3" 
    Example: "trend:fast_ma=10,20;slow_ma=30,50"
    
    Args:
        classifier_specs: List of classifier specification strings
        
    Returns:
        List of classifier configuration dictionaries
    """
    classifiers = []
    
    for spec in classifier_specs:
        try:
            classifier_config = _parse_component_spec(spec, 'classifier')
            if classifier_config:
                classifiers.append(classifier_config)
        except Exception as e:
            logger.error(f"Failed to parse classifier spec '{spec}': {e}")
            continue
    
    logger.info(f"Parsed {len(classifiers)} classifier specifications")
    return classifiers


def _parse_component_spec(spec: str, component_type: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single component specification string.
    
    Args:
        spec: Component specification string
        component_type: Type of component ('strategy' or 'classifier')
        
    Returns:
        Component configuration dictionary
    """
    if ':' not in spec:
        # Simple type without parameters
        return {
            'name': spec,
            'type': spec,
            'params': {}
        }
    
    # Split into type and parameters
    parts = spec.split(':', 1)
    comp_type = parts[0].strip()
    params_str = parts[1].strip()
    
    # Parse parameters
    params = {}
    if params_str:
        param_pairs = params_str.split(';')
        for pair in param_pairs:
            if '=' in pair:
                param_name, param_values_str = pair.split('=', 1)
                param_name = param_name.strip()
                param_values_str = param_values_str.strip()
                
                # Parse parameter values (comma-separated)
                if ',' in param_values_str:
                    # Multiple values - create list
                    values = [_convert_value(v.strip()) for v in param_values_str.split(',')]
                    params[param_name] = values
                else:
                    # Single value
                    params[param_name] = _convert_value(param_values_str)
    
    return {
        'name': comp_type,  # Will be auto-generated during expansion
        'type': comp_type,
        'params': params
    }


def _convert_value(value_str: str) -> Any:
    """
    Convert string value to appropriate Python type.
    
    Args:
        value_str: String value to convert
        
    Returns:
        Converted value (int, float, bool, or str)
    """
    value_str = value_str.strip()
    
    # Try boolean
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Try integer
    try:
        if '.' not in value_str:
            return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def load_parameters_from_file(parameters_file: str) -> Dict[str, Any]:
    """
    Load parameters from a JSON file (e.g., from analytics export).
    
    Args:
        parameters_file: Path to JSON parameters file
        
    Returns:
        Loaded parameters dictionary
        
    Raises:
        FileNotFoundError: If parameters file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    params_path = Path(parameters_file)
    
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {parameters_file}")
    
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        logger.info(f"Loaded parameters from {parameters_file}")
        return params
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in parameters file {parameters_file}: {e}")


def build_config_from_cli(args) -> Dict[str, Any]:
    """
    Build complete configuration from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Configuration dictionary for topology execution
    """
    config = {}
    
    # Parse strategy specifications
    if args.strategies:
        strategies = parse_strategy_specs(args.strategies)
        if strategies:
            config['strategies'] = strategies
    
    # Parse classifier specifications  
    if args.classifiers:
        classifiers = parse_classifier_specs(args.classifiers)
        if classifiers:
            config['classifiers'] = classifiers
    
    # Load parameters from file if specified
    if args.parameters:
        try:
            loaded_params = load_parameters_from_file(args.parameters)
            # Merge loaded parameters with CLI-specified ones
            if 'strategies' in loaded_params:
                if 'strategies' in config:
                    # CLI strategies take precedence, but merge if possible
                    logger.warning("Both CLI and file strategies specified. Using CLI strategies.")
                else:
                    config['strategies'] = loaded_params['strategies']
            
            if 'classifiers' in loaded_params:
                if 'classifiers' in config:
                    logger.warning("Both CLI and file classifiers specified. Using CLI classifiers.")
                else:
                    config['classifiers'] = loaded_params['classifiers']
                    
        except Exception as e:
            logger.error(f"Failed to load parameters from file: {e}")
            raise
    
    # Add default configuration if nothing specified
    if not config.get('strategies') and not config.get('classifiers'):
        logger.warning("No strategies or classifiers specified. Using minimal default configuration.")
        config['strategies'] = [
            {
                'name': 'momentum_default',
                'type': 'momentum',
                'params': {
                    'lookback': 20,
                    'threshold': 0.01
                }
            }
        ]
    
    # Add basic execution configuration for signal generation
    if 'execution' not in config:
        config['execution'] = {
            'enable_event_tracing': True,
            'trace_settings': {
                'use_sparse_storage': True,
                'storage': {
                    'base_dir': './workspaces'
                }
            }
        }
    
    # Add default symbols if not specified
    if 'symbols' not in config:
        config['symbols'] = ['SPY']
    
    return config