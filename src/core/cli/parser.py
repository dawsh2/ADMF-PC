"""
Command-line argument parser for ADMF-PC.

Provides structured argument parsing with proper type annotations
and separation of concerns from main application logic.

Includes parameter parsing for strategy and classifier specifications.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CLIArgs:
    """Structured representation of CLI arguments."""
    # Core arguments
    config: Optional[str] = None
    
    # Clean topology action flags (mutually exclusive)
    signal_generation: bool = False
    backtest: bool = False
    signal_replay: bool = False
    optimize: bool = False
    alpaca: bool = False
    universal: bool = False
    
    # Workflow and sequence arguments  
    workflow: Optional[str] = None
    sequence: Optional[str] = None
    
    # Data arguments
    dataset: Optional[str] = None
    bars: Optional[int] = None
    split_ratio: Optional[float] = None
    
    # Execution arguments
    parallel: Optional[int] = None
    checkpoint: Optional[str] = None
    output_dir: Optional[str] = None
    close_eod: bool = False
    force: bool = False
    
    # Study and WFV arguments
    results_dir: Optional[str] = None
    wfv_windows: Optional[int] = None
    wfv_window: Optional[int] = None
    phase: Optional[str] = None
    
    # Strategy parameter arguments
    strategies: Optional[List[str]] = None
    classifiers: Optional[List[str]] = None
    parameters: Optional[str] = None
    
    # Logging arguments
    log_level: str = 'INFO'
    log_events: List[str] = None
    log_file: Optional[str] = None
    log_json: bool = False
    verbose: bool = False
    
    # Development arguments
    dry_run: bool = False
    profile: bool = False
    schema_docs: bool = False
    
    # Strategy discovery arguments
    list_strategies: bool = False
    strategy_filter: Optional[str] = None
    
    # Notebook generation arguments
    notebook: Optional[str] = None  # Path to existing results
    launch_notebook: bool = False
    notebook_template: Optional[str] = None
    auto_notebook: bool = True  # Auto-generate notebook after runs

    def __post_init__(self):
        if self.log_events is None:
            self.log_events = []


def parse_arguments() -> CLIArgs:
    """Parse command line arguments and return structured CLIArgs object."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Decision Making Framework - Protocol Components'
    )
    
    # Core arguments
    parser.add_argument(
        '--config',
        type=str,
        required=False,  # Made optional to allow --schema-docs without config
        help='Path to configuration file (YAML)'
    )
    
    # Clean topology action flags - each implies a specific topology
    action_group = parser.add_mutually_exclusive_group()
    
    action_group.add_argument(
        '--signal-generation', '-sg',
        action='store_true',
        help='Generate trading signals from strategies'
    )
    
    action_group.add_argument(
        '--backtest', '-bt',
        action='store_true',
        help='Run backtest simulation'
    )
    
    action_group.add_argument(
        '--signal-replay', '-sr',
        action='store_true',
        help='Replay previously generated signals'
    )
    
    action_group.add_argument(
        '--alpaca', '-a',
        action='store_true',
        help='Run live trading with Alpaca WebSocket data'
    )
    
    action_group.add_argument(
        '--universal', '-u',
        action='store_true',
        help='Use universal topology with complete trading pipeline (signals, portfolio, execution)'
    )
    
    # Workflow for complex multi-phase executions
    action_group.add_argument(
        '--workflow', '-w',
        type=str,
        help='Execute a workflow pattern (e.g., research_pipeline)'
    )
    
    # Sequence patterns
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        help='Apply sequence pattern (e.g., walk_forward, parameter_sweep)'
    )
    
    # Optimization flag (not mutually exclusive - controls parameter expansion)
    parser.add_argument(
        '--optimize', '-opt',
        action='store_true',
        help='Run parameter optimization (expands parameter_space from config)'
    )
    
    # List strategies
    parser.add_argument(
        '--list-strategies',
        action='store_true',
        help='List available strategies and exit'
    )
    
    parser.add_argument(
        '--strategy-filter',
        type=str,
        help='Filter strategies by category (e.g., oscillator, volatility, structure)'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'test', 'full'],
        default=None,
        help='Dataset to use (enables reproducible train/test splits)'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Limit data to first N bars (useful for testing)'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=None,
        help='Train/test split ratio when dataset is "full"'
    )
    
    # Execution arguments
    parser.add_argument(
        '--parallel',
        type=int,
        default=None,
        help='Number of parallel workers for optimization'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Resume from checkpoint file'
    )
    
    parser.add_argument(
        '--close-eod',
        action='store_true',
        help='Force close all positions at end of day (prevents overnight holding)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation of strategies even if they already exist in traces'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output files'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-events',
        type=str,
        nargs='*',
        choices=['BAR', 'INDICATOR', 'SIGNAL', 'ORDER', 'FILL', 'PORTFOLIO', 'TRADE_LOOP'],
        default=[],
        help='Enable detailed logging for specific event types'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log to file instead of console'
    )
    
    parser.add_argument(
        '--log-json',
        action='store_true',
        help='Use structured JSON logging format'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (equivalent to --log-level DEBUG)'
    )
    
    # Study and WFV arguments
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Study-level directory name for organizing related WFV runs'
    )
    
    parser.add_argument(
        '--wfv-windows',
        type=int,
        default=None,
        help='Total number of walk-forward validation windows'
    )
    
    parser.add_argument(
        '--wfv-window',
        type=int,
        default=None,
        help='Execute specific WFV window (1-based index)'
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['train', 'test'],
        default=None,
        help='Execution phase for walk-forward validation (train or test)'
    )
    
    # Strategy parameter arguments for config-less operation
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        help='Strategy specifications: "type:param1=val1,val2;param2=val3" '
             '(e.g., "momentum:lookback=10,20,30;threshold=0.01,0.02")',
        default=None
    )
    
    parser.add_argument(
        '--classifiers',
        type=str,
        nargs='+',
        help='Classifier specifications: "type:param1=val1,val2;param2=val3" '
             '(e.g., "trend:fast_ma=10,20;slow_ma=30,50")',
        default=None
    )
    
    parser.add_argument(
        '--parameters',
        type=str,
        help='Load parameters from analytics export (JSON file path)',
        default=None
    )
    
    # Development arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    parser.add_argument(
        '--schema-docs',
        action='store_true',
        help='Print configuration schema documentation and exit'
    )
    
    # Notebook generation arguments
    parser.add_argument(
        '--notebook',
        type=str,
        help='Generate analysis notebook from existing results (e.g., --notebook config/bollinger/results/latest)'
    )
    
    parser.add_argument(
        '--launch-notebook',
        action='store_true',
        help='Execute notebook and launch Jupyter (implies --notebook, requires papermill)'
    )
    
    parser.add_argument(
        '--notebook-template',
        type=str,
        help='Use specific notebook template (optional)')
    
    parser.add_argument(
        '--no-auto-notebook',
        action='store_true',
        help='Disable automatic notebook generation after runs'
    )
    
    args = parser.parse_args()
    
    # Convert to structured CLIArgs
    return CLIArgs(
        config=args.config,
        # Clean topology action flags
        signal_generation=getattr(args, 'signal_generation', False),
        backtest=getattr(args, 'backtest', False),
        signal_replay=getattr(args, 'signal_replay', False),
        optimize=getattr(args, 'optimize', False),
        alpaca=getattr(args, 'alpaca', False),
        universal=getattr(args, 'universal', False),
        # Workflow and sequence arguments
        workflow=getattr(args, 'workflow', None),
        sequence=getattr(args, 'sequence', None),
        # Data arguments
        dataset=args.dataset,
        bars=args.bars,
        split_ratio=getattr(args, 'split_ratio', None),
        close_eod=args.close_eod,
        force=getattr(args, 'force', False),
        # Execution arguments
        parallel=args.parallel,
        checkpoint=args.checkpoint,
        output_dir=getattr(args, 'output_dir', None),
        # Logging arguments
        log_level=getattr(args, 'log_level', 'INFO'),
        log_events=getattr(args, 'log_events', []),
        log_file=getattr(args, 'log_file', None),
        log_json=getattr(args, 'log_json', False),
        verbose=args.verbose,
        # Study and WFV arguments
        results_dir=getattr(args, 'results_dir', None),
        wfv_windows=getattr(args, 'wfv_windows', None),
        wfv_window=getattr(args, 'wfv_window', None),
        phase=getattr(args, 'phase', None),
        # Strategy parameter arguments
        strategies=getattr(args, 'strategies', None),
        classifiers=getattr(args, 'classifiers', None),
        parameters=getattr(args, 'parameters', None),
        # Development arguments
        dry_run=getattr(args, 'dry_run', False),
        profile=args.profile,
        schema_docs=getattr(args, 'schema_docs', False),
        # Strategy discovery arguments
        list_strategies=getattr(args, 'list_strategies', False),
        strategy_filter=getattr(args, 'strategy_filter', None),
        # Notebook generation arguments
        notebook=args.notebook,
        launch_notebook=getattr(args, 'launch_notebook', False),
        notebook_template=getattr(args, 'notebook_template', None),
        auto_notebook=not getattr(args, 'no_auto_notebook', False)
    )


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