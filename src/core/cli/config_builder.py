"""
Workflow configuration builder for ADMF-PC.

Handles the merging of CLI arguments with YAML configuration files
to produce WorkflowConfig objects for execution.
"""

from typing import Dict, Any
import yaml
from pathlib import Path

from .parser import CLIArgs
from ..types.workflow import WorkflowConfig, WorkflowType


def build_workflow_config(args: CLIArgs, base_config: Dict[str, Any]) -> WorkflowConfig:
    """
    Build workflow configuration from CLI arguments and base YAML config.
    
    Args:
        args: Parsed CLI arguments
        base_config: Loaded YAML configuration
        
    Returns:
        WorkflowConfig object ready for execution
    """
    # Determine execution mode
    mode = args.mode or base_config.get('workflow_type', 'backtest')
    
    # Extract strategies from base config and add to backtest config
    backtest_config = base_config.get('backtest', {}).copy()
    if 'strategies' in base_config:
        backtest_config['strategies'] = base_config['strategies']
        
    # Create workflow config
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType(mode if mode in ['backtest', 'optimization', 'live'] else 'backtest'),
        parameters=base_config,  # Store entire YAML config for access to all sections
        data_config=backtest_config.get('data', base_config.get('data', {})),  # Try backtest.data first, then top-level data
        backtest_config=backtest_config,
        optimization_config=base_config.get('optimization', {}),
        analysis_config=base_config.get('analysis', {})
    )
    
    # Apply CLI overrides to data config
    if args.dataset:
        workflow_config.data_config['dataset'] = args.dataset
    if args.bars:
        workflow_config.data_config['max_bars'] = args.bars
    if args.split_ratio:
        workflow_config.data_config['split_ratio'] = args.split_ratio
    
    # Apply CLI overrides to parameters
    if args.parallel:
        workflow_config.parameters['parallel_workers'] = args.parallel
    if args.checkpoint:
        workflow_config.parameters['checkpoint_file'] = args.checkpoint
    if args.output_dir:
        workflow_config.parameters['output_dir'] = args.output_dir
        
    # Apply logging and development options
    if args.verbose:
        workflow_config.parameters['verbose'] = True
    if args.log_file:
        workflow_config.parameters['log_file'] = args.log_file
    if args.dry_run:
        workflow_config.parameters['dry_run'] = True
    if args.profile:
        workflow_config.parameters['profile'] = True
    
    return workflow_config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")


def build_workflow_from_cli(args: CLIArgs) -> WorkflowConfig:
    """
    Build complete workflow configuration from CLI arguments.
    
    This is a convenience function that loads the YAML config
    and merges it with CLI arguments in one step.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        WorkflowConfig ready for execution
    """
    base_config = load_yaml_config(args.config)
    return build_workflow_config(args, base_config)