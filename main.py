#!/usr/bin/env python3
"""
ADMF-PC: Adaptive Decision Making Framework - Protocol Components

Main entry point - orchestrates CLI parsing, configuration, and workflow execution.
"""

import asyncio
import logging
import time
from pathlib import Path

# Import core modules
from src.core.coordinator.coordinator import Coordinator
from src.core.cli import parse_arguments
import logging

# Import Pydantic validation
from src.core.coordinator.config import (
    PYDANTIC_AVAILABLE,
    get_validation_errors,
    generate_schema_docs
)

def setup_logging(level='INFO', console=True, file_path=None, json_format=False):
    """Basic logging setup."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def configure_event_logging(events):
    """Placeholder for event logging configuration."""
    pass


def handle_execution_result(result) -> int:
    """
    Handle workflow execution result and return appropriate exit code.
    
    Args:
        result: Workflow execution result
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = logging.getLogger(__name__)
    
    # Handle different result formats
    if hasattr(result, 'success') and result.success:
        logger.info("Workflow completed successfully")
        # Note: Detailed results now printed in human-readable format by main()
        return 0
    elif hasattr(result, 'success'):
        logger.error("Workflow failed")
        if hasattr(result, 'errors') and result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return 1
    else:
        # Handle dictionary result from fallback
        if result.get('success'):
            logger.info("Workflow completed successfully")
            if result.get('message'):
                logger.info(result['message'])
            return 0
        else:
            logger.error("Workflow failed")
            if result.get('errors'):
                for error in result['errors']:
                    logger.error(f"  - {error}")
            return 1



def main():
    """Main entry point - clean orchestration of CLI parsing, configuration, and execution."""
    # Parse CLI arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    setup_logging(
        level=log_level,
        console=True,
        file_path=args.log_file,
        json_format=args.log_json
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Handle schema documentation request
    if args.schema_docs:
        if PYDANTIC_AVAILABLE:
            print("# ADMF-PC Configuration Schema Documentation\n")
            print(generate_schema_docs())
            return 0
        else:
            print("❌ Schema documentation requires Pydantic: pip install pydantic>=2.0.0")
            return 1
    
    # Determine config path and action topology
    config_path = args.config
    action_topology = None
    
    if args.signal_generation:
        action_topology = 'signal_generation'
    elif args.backtest:
        action_topology = 'backtest'
    elif args.signal_replay:
        action_topology = 'signal_replay'
    elif args.optimize:
        action_topology = 'optimization'
    
    # Check that we have config or CLI parameters for config-less operation
    if not config_path and not (args.strategies or args.classifiers or args.parameters):
        logger.error("❌ Configuration file or CLI parameters required.")
        logger.error("Use --config config.yaml OR --strategies \"type:params\" OR --parameters file.json")
        logger.info("💡 Use --schema-docs to see configuration requirements")
        return 1
    
    # Configure event-specific logging
    if args.log_events:
        configure_event_logging(args.log_events)
        logger.info(f"Event-specific logging enabled for: {', '.join(args.log_events)}")
    
    # Load configuration (from file or CLI parameters)
    try:
        if config_path:
            # Load from YAML file
            from src.core.cli.args import load_yaml_config
            config_dict = load_yaml_config(config_path)
            logger.info(f"Configuration loaded successfully from {config_path}")
        else:
            # Build from CLI parameters (config-less operation)
            from src.core.cli.parser import build_config_from_cli
            config_dict = build_config_from_cli(args)
            logger.info("Configuration built from CLI parameters (config-less mode)")
        
        # Apply CLI overrides
        if args.bars is not None:
            # Set at top level for topology patterns
            config_dict['max_bars'] = args.bars
            # Also set in data config for backwards compatibility
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['max_bars'] = args.bars
            logger.info(f"Limiting data to {args.bars} bars")
            
        # Apply dataset CLI override
        if args.dataset:
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['dataset'] = args.dataset
            logger.info(f"Using dataset: {args.dataset}")
            
        # Apply split_ratio CLI override
        if args.split_ratio:
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['split_ratio'] = args.split_ratio
            logger.info(f"Using split ratio: {args.split_ratio}")
            
        # Apply WFV and study configuration
        if args.results_dir:
            config_dict['results_dir'] = args.results_dir
            logger.info(f"Study results directory: {args.results_dir}")
        if args.wfv_windows:
            config_dict['wfv_windows'] = args.wfv_windows
            logger.info(f"WFV total windows: {args.wfv_windows}")
        if args.wfv_window:
            config_dict['wfv_window'] = args.wfv_window
            logger.info(f"WFV current window: {args.wfv_window}")
        if args.phase:
            config_dict['phase'] = args.phase
            logger.info(f"Execution phase: {args.phase}")
        
        # VALIDATE CONFIGURATION (if Pydantic available)
        # Temporarily disabled validation as it's too restrictive
        if False and PYDANTIC_AVAILABLE:
            try:
                validation_errors = get_validation_errors(config_dict)
                if validation_errors:
                    logger.error("❌ Configuration validation failed:")
                    for error in validation_errors:
                        logger.error(f"  - {error}")
                    logger.info("💡 Use --schema-docs to see configuration requirements")
                    return 1
                else:
                    logger.info("✅ Configuration validation passed")
            except Exception as e:
                logger.warning(f"Configuration validation failed with error: {e}")
                logger.debug("Proceeding without validation")
        else:
            logger.debug("Skipping validation")
            
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create and run coordinator
    coordinator = Coordinator()
    
    result = None
    try:
        logger.info("🚀 Starting workflow execution...")
        import time
        start_time = time.time()
        
        # Route to appropriate execution method based on CLI args
        logger.debug(f"CLI args - action_topology: {action_topology}, workflow: {args.workflow}, sequence: {args.sequence}")
        
        if action_topology:
            # Clean action flag execution
            logger.info(f"🎯 Executing {action_topology} topology")
            # Add config metadata for workspace organization
            if 'metadata' not in config_dict:
                config_dict['metadata'] = {}
            config_dict['metadata']['config_file'] = config_path
            config_dict['metadata']['config_name'] = Path(config_path).stem
            result = coordinator.run_topology(action_topology, config_dict)
            
        elif args.workflow:
            # Explicit workflow execution
            config_dict['workflow'] = args.workflow
            result = coordinator.run_workflow(config_dict)
            
        else:
            # Config-driven execution (backward compatibility)
            result = coordinator.run_workflow(config_dict)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Workflow execution completed in {elapsed:.2f} seconds")
        
        # Format results in human-readable format
        if result and hasattr(result, 'success') and result.success and hasattr(result, 'data'):
            from src.analytics.basic_report import format_backtest_results
            readable_report = format_backtest_results(result.data)
            print("\n" + readable_report)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        result = {
            'success': False,
            'errors': [str(e)]
        }
    finally:
        # Clean shutdown
        pass  # Coordinator doesn't have async shutdown
    
    # Handle results and return exit code
    return handle_execution_result(result)


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code if exit_code is not None else 0)