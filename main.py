#!/usr/bin/env python3
"""
ADMF-PC: Adaptive Decision Making Framework - Protocol Components

Main entry point - orchestrates CLI parsing, configuration, and workflow execution.
"""

import asyncio
import logging
import time

# Import core modules
from src.core.coordinator.coordinator import Coordinator
from src.core.cli import parse_arguments
# TODO: Fix logging import - utils.logging module not found
# from src.core.utils.logging import setup_logging, configure_event_logging
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
    
    # Check that config is provided for normal operations
    if not args.config:
        logger.error("❌ --config is required for workflow execution")
        logger.info("💡 Use --schema-docs to see configuration requirements")
        return 1
    
    # Configure event-specific logging
    if args.log_events:
        configure_event_logging(args.log_events)
        logger.info(f"Event-specific logging enabled for: {', '.join(args.log_events)}")
    
    # Load raw YAML configuration
    try:
        from src.core.cli.args import load_yaml_config
        config_dict = load_yaml_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Apply CLI overrides
        if args.bars is not None:
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['max_bars'] = args.bars
            logger.info(f"Limiting data to {args.bars} bars")
        
        # VALIDATE CONFIGURATION (if Pydantic available)
        if PYDANTIC_AVAILABLE:
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
            logger.debug("Pydantic validation not available, skipping validation")
            
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
        
        # Pass the raw config dict to coordinator
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