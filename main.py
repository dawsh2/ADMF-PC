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
from src.core.cli.config_builder import build_workflow_from_cli
from src.core.utils.logging import setup_logging, configure_event_logging


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



async def main():
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
    
    # Configure event-specific logging
    if args.log_events:
        configure_event_logging(args.log_events)
        logger.info(f"Event-specific logging enabled for: {', '.join(args.log_events)}")
    
    # Build workflow configuration from CLI + YAML
    try:
        workflow_config = build_workflow_from_cli(args)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create and run coordinator
    coordinator = Coordinator(
        enable_communication=True,
        enable_yaml=True
    )
    
    result = None
    try:
        logger.info("🚀 Starting workflow execution...")
        import time
        start_time = time.time()
        
        result = await coordinator.execute_workflow(
            workflow_config
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Workflow execution completed in {elapsed:.2f} seconds")
        
        # Format results in human-readable format
        if result and hasattr(result, 'success') and result.success and hasattr(result, 'data'):
            from src.analytics.basic_report import format_backtest_results
            readable_report = format_backtest_results(result.data)
            print("\n" + readable_report)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        result = {
            'success': False,
            'errors': [str(e)]
        }
    finally:
        # Clean shutdown
        await coordinator.shutdown()
    
    # Handle results and return exit code
    return handle_execution_result(result)


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code if exit_code is not None else 0)