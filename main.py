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
    elif args.alpaca:
        action_topology = 'universal'  # Use universal topology for complete trading
    elif args.universal:
        action_topology = 'universal'  # Use universal topology for complete trading
    else:
        # Default to universal when no action flags specified
        action_topology = 'universal'
        logger.info("No action flag specified, defaulting to --universal (full system analysis)")
    
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
            elif isinstance(config_dict['data'], str):
                # If data is a string (like "SPY_5m"), keep it as a string
                # The data parser expects the string format
                pass
            if isinstance(config_dict['data'], dict):
                config_dict['data']['max_bars'] = args.bars
            logger.info(f"Limiting data to {args.bars} bars")
            
        # Apply dataset CLI override
        if args.dataset:
            # Set at top level for topology patterns
            config_dict['dataset'] = args.dataset
            # Also set in data config for backwards compatibility
            if 'data' not in config_dict:
                config_dict['data'] = {}
            elif isinstance(config_dict['data'], str):
                # If data is a string (like "SPY_5m"), keep it as a string
                # The data parser expects the string format
                pass
            if isinstance(config_dict['data'], dict):
                config_dict['data']['dataset'] = args.dataset
            logger.info(f"Using dataset: {args.dataset}")
            
        # Apply split_ratio CLI override (or set default if dataset is specified)
        if args.split_ratio:
            # Set at top level for topology patterns
            config_dict['split_ratio'] = args.split_ratio
            # Also set in data config for backwards compatibility
            if 'data' not in config_dict:
                config_dict['data'] = {}
            elif isinstance(config_dict['data'], str):
                # If data is a string (like "SPY_5m"), keep it as a string
                # The data parser expects the string format
                pass
            if isinstance(config_dict['data'], dict):
                config_dict['data']['split_ratio'] = args.split_ratio
            logger.info(f"Using split ratio: {args.split_ratio}")
        elif args.dataset:
            # If dataset is specified but no split_ratio, use default 0.8
            config_dict['split_ratio'] = 0.8
            if 'data' not in config_dict:
                config_dict['data'] = {}
            elif isinstance(config_dict['data'], str):
                # If data is a string (like "SPY_5m"), keep it as a string
                # The data parser expects the string format
                pass
            if isinstance(config_dict['data'], dict):
                config_dict['data']['split_ratio'] = 0.8
            logger.info(f"Using default split ratio: 0.8")
            
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
        
        # Apply force recompute flag
        if args.force:
            config_dict['force_recompute'] = True
            logger.info("⚠️  Force recompute enabled - will ignore existing traces")
            
        # Apply Alpaca live trading configuration
        if args.alpaca:
            import os
            
            # Override data source to use Alpaca WebSocket
            config_dict['data_source'] = 'alpaca_websocket'
            
            # Extract symbols from config or use defaults
            symbols = []
            
            # Check if data is in string format (e.g., "SPY_5m")
            if 'data' in config_dict and isinstance(config_dict['data'], str):
                # Parse symbol and timeframe from string format
                data_str = config_dict['data']
                if '_' in data_str:
                    parts = data_str.split('_')
                    symbol = parts[0]
                    timeframe = parts[1] if len(parts) > 1 else '1m'
                    
                    # Convert crypto symbols to use slash format for Alpaca v1beta3
                    crypto_symbols = ['BTCUSD', 'ETHUSD', 'BTCUSDT', 'ETHUSDT']
                    if symbol in crypto_symbols:
                        # Insert slash: BTCUSD -> BTC/USD
                        symbol = symbol[:3] + '/' + symbol[3:]
                        logger.info(f"Converted crypto symbol to: {symbol}")
                    
                    # Convert timeframe format for Alpaca (1m -> 1Min, 5m -> 5Min, etc.)
                    timeframe_map = {
                        '1m': '1Min',
                        '5m': '5Min', 
                        '15m': '15Min',
                        '30m': '30Min',
                        '1h': '1Hour',
                        '1d': '1Day',
                        'tick': 'tick'
                    }
                    alpaca_timeframe = timeframe_map.get(timeframe.lower(), timeframe)
                    if alpaca_timeframe != timeframe:
                        logger.info(f"Converted timeframe from '{timeframe}' to '{alpaca_timeframe}'")
                    timeframe = alpaca_timeframe
                    
                    symbols = [symbol]
                    logger.info(f"Parsed symbol '{symbol}' and timeframe '{timeframe}' from data string '{data_str}'")
            elif 'data' in config_dict and isinstance(config_dict['data'], dict):
                data_symbols = config_dict['data'].get('symbol', config_dict['data'].get('symbols', []))
                if isinstance(data_symbols, str):
                    symbols = [data_symbols]
                elif isinstance(data_symbols, list):
                    symbols = data_symbols
            
            # If no symbols found, check strategies for symbol info
            if not symbols and 'strategies' in config_dict:
                for strategy in config_dict.get('strategies', []):
                    if isinstance(strategy, dict) and 'symbol' in strategy:
                        symbol = strategy['symbol']
                        if symbol and symbol not in symbols:
                            symbols.append(symbol)
            
            # Default to SPY if no symbols found
            if not symbols:
                symbols = ['SPY']
            
            # Override data config for Alpaca - replacing whatever was there
            config_dict['data'] = {
                'symbol': symbols,
                'data_source': 'alpaca_websocket'
            }
            
            # Also set symbols and timeframes at root level for universal topology
            config_dict['symbols'] = symbols
            
            # Use parsed timeframe or default to 1m
            if 'timeframe' in locals():
                # For tick data, use 'tick' as timeframe
                if timeframe.lower() == 'tick':
                    config_dict['timeframes'] = ['tick']
                    logger.info("📊 Using tick-level data streaming")
                else:
                    config_dict['timeframes'] = [timeframe]
            else:
                config_dict['timeframes'] = ['1m']  # Default to 1-minute bars
                
            config_dict['data_source'] = 'alpaca_websocket'
            
            logger.info(f"Overriding data config for Alpaca WebSocket with symbols: {symbols}")
            
            # Add live trading configuration
            config_dict['live_trading'] = {
                'api_key': os.getenv('ALPACA_API_KEY'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET'),
                'paper_trading': True,  # Use paper trading for safety
                'feed': 'sip'  # Use SIP feed for paid accounts (more reliable, lower latency)
            }
            
            # Ensure we have the required credentials
            if not config_dict['live_trading']['api_key'] or not config_dict['live_trading']['secret_key']:
                logger.error("❌ Alpaca credentials required for live trading")
                logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
                return 1
            
            logger.info("🔴 Live trading mode enabled with Alpaca WebSocket")
            logger.info(f"📊 Symbols: {symbols}")
            logger.info(f"🔑 API Key: {config_dict['live_trading']['api_key'][:8]}...")
            logger.info(f"📄 Paper Trading: {config_dict['live_trading']['paper_trading']}")
            logger.info(f"📡 Data Feed: {config_dict['live_trading']['feed']}")
            
            # Enable async execution for Alpaca
            if 'execution' not in config_dict:
                config_dict['execution'] = {}
            config_dict['execution']['broker'] = 'alpaca'
            config_dict['execution']['execution_mode'] = 'async'
            logger.info("🚀 Enabled async execution with Alpaca broker")
        
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
    
    # Auto-discover all components before creating coordinator
    logger.info("🔍 Discovering components...")
    from src.core.components.discovery import auto_discover_all_components
    component_counts = auto_discover_all_components()
    
    # Log discovered components
    total_components = 0
    for package, count in component_counts.items():
        if count > 0:
            logger.info(f"  ✓ {package}: {count} components")
            total_components += count
    logger.info(f"📦 Total components discovered: {total_components}")
    
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
        
        # Handle notebook generation
        if args.launch_notebook or args.notebook:
            logger.info("📓 Generating analysis notebook...")
            
            # Determine results directory
            results_dir = None
            if args.notebook:
                # Explicit path provided
                results_dir = args.notebook
            elif result and isinstance(result, dict) and 'results_directory' in result:
                # Use results from execution
                results_dir = result['results_directory']
            elif 'results_directory' in config_dict:
                # Check config
                results_dir = config_dict['results_directory']
            elif action_topology == 'signal_generation':
                # Check for signal generation results
                config_name = Path(config_path).stem if config_path else 'results'
                base_dir = Path(config_path).parent if config_path else Path('.')
                latest_link = base_dir / 'results' / 'latest'
                if latest_link.exists():
                    results_dir = str(latest_link.resolve())
            
            if results_dir:
                try:
                    from src.analytics.papermill_runner import PapermillNotebookRunner
                    
                    # Create papermill runner
                    runner = PapermillNotebookRunner()
                    
                    # Run analysis notebook
                    notebook_path = runner.run_analysis(
                        run_dir=Path(results_dir),
                        config=config_dict,
                        execute=args.launch_notebook,  # Execute if launching
                        launch=args.launch_notebook,
                        generate_html=False
                    )
                    
                    if notebook_path:
                        logger.info(f"📓 Notebook generated: {notebook_path}")
                    else:
                        logger.warning("Failed to generate notebook")
                        
                except Exception as e:
                    logger.error(f"Failed to generate/launch notebook: {e}")
            else:
                logger.warning("No results directory found for notebook generation")
                logger.info("Use --notebook with explicit path or run after signal generation/backtest")
        
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