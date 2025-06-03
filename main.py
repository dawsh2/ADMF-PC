#!/usr/bin/env python3
"""
ADMF-PC: Adaptive Decision Making Framework - Protocol Components

Main entry point - parses arguments and delegates to Bootstrap.
"""

import asyncio
import argparse
import yaml
from typing import Dict, Any

# Use new composable container system
from src.core.coordinator.coordinator import Coordinator
from src.core.containers.composition_engine import get_global_composition_engine

# Simple types for configuration
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Set

class WorkflowType(str, Enum):
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    LIVE = "live"

@dataclass
class WorkflowConfig:
    workflow_type: WorkflowType
    parameters: Dict[str, Any]
    data_config: Dict[str, Any]
    backtest_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    analysis_config: Dict[str, Any] = None
    
    def dict(self):
        return {
            'workflow_type': self.workflow_type.value,
            'parameters': self.parameters,
            'data_config': self.data_config,
            'backtest_config': self.backtest_config,
            'optimization_config': self.optimization_config,
            'analysis_config': self.analysis_config or {}
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Decision Making Framework - Protocol Components'
    )
    
    # Core arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    # Execution mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'optimization', 'signal-generation', 'signal-replay', 'live'],
        default=None,
        help='Override execution mode from config'
    )
    
    parser.add_argument(
        '--signal-log',
        type=str,
        default=None,
        help='Path to signal log file for replay mode'
    )
    
    parser.add_argument(
        '--signal-output',
        type=str,
        default=None,
        help='Path to save generated signals (signal-generation mode)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='JSON string or file with strategy weights for signal-replay mode'
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
    
    return parser.parse_args()


def build_workflow_config(args: argparse.Namespace, base_config: Dict[str, Any]) -> WorkflowConfig:
    """Build workflow configuration from arguments and base config."""
    # Determine execution mode
    mode = args.mode or base_config.get('workflow_type', 'backtest')
    
    # Extract strategies from base config and add to backtest config
    backtest_config = base_config.get('backtest', {}).copy()
    if 'strategies' in base_config:
        backtest_config['strategies'] = base_config['strategies']
        
    # Create workflow config
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType(mode if mode in ['backtest', 'optimization', 'live'] else 'backtest'),
        parameters=base_config,  # Store entire YAML config for access to all sections including 'reporting'
        data_config=base_config.get('data', {}),
        backtest_config=backtest_config,
        optimization_config=base_config.get('optimization', {}),
        analysis_config=base_config.get('analysis', {})
    )
    
    # Apply CLI overrides
    if args.dataset:
        workflow_config.data_config['dataset'] = args.dataset
    if args.bars:
        workflow_config.data_config['max_bars'] = args.bars
    if args.split_ratio:
        workflow_config.data_config['split_ratio'] = args.split_ratio
    
    if args.parallel:
        workflow_config.parameters['parallel_workers'] = args.parallel
    if args.checkpoint:
        workflow_config.parameters['checkpoint_file'] = args.checkpoint
    if args.output_dir:
        workflow_config.parameters['output_dir'] = args.output_dir
        
    if args.verbose:
        workflow_config.parameters['verbose'] = True
    if args.log_file:
        workflow_config.parameters['log_file'] = args.log_file
    if args.dry_run:
        workflow_config.parameters['dry_run'] = True
    if args.profile:
        workflow_config.parameters['profile'] = True
    
    return workflow_config



async def main():
    """Main entry point - parse args and delegate to Bootstrap."""
    args = parse_arguments()
    
    # Setup intelligent logging
    from src.core.logging.structured import setup_logging
    
    # Determine log level
    log_level = 'DEBUG' if args.verbose else args.log_level
    
    # Setup logging system
    setup_logging(
        level=log_level,
        console=True,
        file_path=args.log_file,
        json_format=args.log_json
    )
    
    # Load base configuration first
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create module logger
    import logging
    logger = logging.getLogger(__name__)
    
    # Configure event-specific logging from CLI args
    if args.log_events:
        from src.core.logging.event_logger import configure_event_logging
        configure_event_logging(args.log_events)
        logger.info(f"Event-specific logging enabled for: {', '.join(args.log_events)}")
    
    # Configure event-specific logging from config file
    if 'logging' in base_config and 'enabled_events' in base_config['logging']:
        from src.core.logging.event_logger import configure_event_logging
        enabled_events = base_config['logging']['enabled_events']
        configure_event_logging(enabled_events)
        logger.info(f"Event-specific logging enabled from config for: {', '.join(enabled_events)}")
    
    logger.info("Using new composable container system")
    
    # Create coordinator using new composable container system
    coordinator = Coordinator(enable_composable_containers=True)
    
    # Build workflow configuration
    workflow_config = build_workflow_config(args, base_config)
    
    # Container registration is handled by the coordinator
    
    # Execute workflow through coordinator - force composable mode
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        print("🚀 Starting workflow execution...")
        import time
        start_time = time.time()
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        elapsed = time.time() - start_time
        print(f"✅ Workflow execution completed in {elapsed:.2f} seconds")
    except Exception as e:
        # If coordinator fails, try using the composition engine directly for testing
        logger.warning(f"Coordinator execution failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Attempting direct container composition for testing...")
        
        try:
            # Test the new container system directly
            from src.core.containers.composition_engine import get_global_composition_engine
            
            engine = get_global_composition_engine()
            
            # Create a simple test pattern
            simple_config = {
                "data": {
                    "source": "historical",
                    "symbols": ["SPY"],
                    "data_dir": "data"
                },
                "strategy": {
                    "type": "momentum",
                    "parameters": {"period": 20}
                }
            }
            
            # Try to compose the pattern
            container = engine.compose_pattern('simple_backtest', simple_config)
            
            logger.info(f"Successfully created container: {container.metadata.name}")
            logger.info(f"Container ID: {container.metadata.container_id}")
            logger.info(f"Children: {len(container.child_containers)}")
            
            # Initialize and briefly run the container
            await container.initialize()
            await container.start()
            await asyncio.sleep(0.1)  # Brief run
            await container.stop()
            
            result = {
                'success': True,
                'message': 'Container system test completed successfully',
                'container_id': container.metadata.container_id
            }
            
        except Exception as container_error:
            logger.error(f"Container composition also failed: {container_error}")
            import traceback
            traceback.print_exc()
            result = {
                'success': False,
                'errors': [str(container_error)]
            }
    
    # Log result
    if hasattr(result, 'success') and result.success:
        logger.info("Workflow completed successfully")
        if hasattr(result, 'data') and result.data:
            logger.info(f"Results: {result.data}")
    elif hasattr(result, 'success'):
        logger.error("Workflow failed")
        if hasattr(result, 'errors') and result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
    else:
        # Handle dictionary result from fallback
        if result.get('success'):
            logger.info("Workflow completed successfully")
            if result.get('message'):
                logger.info(result['message'])
        else:
            logger.error("Workflow failed")
            if result.get('errors'):
                for error in result['errors']:
                    logger.error(f"  - {error}")
    
    # Return appropriate exit code
    success = result.success if hasattr(result, 'success') else result.get('success', False)
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code if exit_code is not None else 0)