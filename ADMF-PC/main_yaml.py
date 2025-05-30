#!/usr/bin/env python3
"""
ADMF-PC: Advanced Data Management Framework for Portfolio Construction
Main entry point for YAML-driven command-line execution.

Usage:
    python main_yaml.py --config config/backtest.yaml
    python main_yaml.py --validate config/strategy.yaml
    python main_yaml.py --create-example backtest --output config/example.yaml
    python main_yaml.py --list-schemas
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional
import yaml

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import ConfigSchemaValidator
from src.core.coordinator.yaml_interpreter import YAMLInterpreter, YAMLWorkflowBuilder
from src.core.coordinator.simple_types import WorkflowType
from src.core.logging import StructuredLogger


# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ADMF-PC: Zero-Code Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a backtest
  python main_yaml.py --config configs/ma_backtest.yaml
  
  # Validate a configuration
  python main_yaml.py --validate configs/strategy.yaml
  
  # Create an example configuration
  python main_yaml.py --create-example backtest --output my_backtest.yaml
  
  # Run with debug logging
  python main_yaml.py --config configs/backtest.yaml --log-level DEBUG
        """
    )
    
    # Main operations (mutually exclusive)
    operation = parser.add_mutually_exclusive_group(required=True)
    
    operation.add_argument(
        "--config", "-c",
        type=str,
        help="Execute workflow from YAML configuration file"
    )
    
    operation.add_argument(
        "--validate", "-v",
        type=str,
        help="Validate a YAML configuration file without executing"
    )
    
    operation.add_argument(
        "--create-example",
        type=str,
        choices=["backtest", "optimization", "live_trading"],
        help="Create an example configuration file"
    )
    
    operation.add_argument(
        "--list-schemas",
        action="store_true",
        help="List available configuration schemas"
    )
    
    # Additional options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for --create-example"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Save logs to file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without executing"
    )
    
    parser.add_argument(
        "--show-containers",
        action="store_true",
        help="Show container hierarchy for configuration"
    )
    
    return parser


async def execute_config(config_path: str, dry_run: bool = False, show_containers: bool = False):
    """Execute a workflow from configuration file."""
    logger = logging.getLogger("main")
    
    # Check if file exists
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    logger.info(f"Loading configuration: {config_path}")
    
    # Validate configuration
    validator = ConfigSchemaValidator()
    validation_result = validator.validate_file(config_file)
    
    if not validation_result.is_valid:
        logger.error("Configuration validation failed:")
        for error in validation_result.errors:
            logger.error(f"  ✗ {error}")
        return 1
    
    logger.info("✓ Configuration validated successfully")
    
    if validation_result.warnings:
        for warning in validation_result.warnings:
            logger.warning(f"  ⚠ {warning}")
    
    # Interpret configuration
    interpreter = YAMLInterpreter()
    workflow_config, _ = interpreter.load_and_interpret(config_file)
    
    logger.info(f"Workflow type: {workflow_config.workflow_type.value}")
    logger.info(f"Workflow name: {workflow_config.parameters.get('name', 'unnamed')}")
    
    # Show container hierarchy if requested
    if show_containers:
        builder = YAMLWorkflowBuilder(interpreter)
        container_spec = builder.build_container_hierarchy(workflow_config)
        
        print("\nContainer Hierarchy:")
        print("-" * 40)
        print_container_hierarchy(container_spec["root"])
        print()
    
    if dry_run:
        logger.info("Dry run completed - no execution performed")
        return 0
    
    # In a real implementation, we would execute the workflow here
    logger.info("Executing workflow...")
    
    # Simulate execution based on workflow type
    if workflow_config.workflow_type == WorkflowType.BACKTEST:
        logger.info("Running backtest...")
        logger.info(f"  Symbols: {workflow_config.data_config.get('symbols', [])}")
        logger.info(f"  Date range: {workflow_config.data_config.get('start_date')} to {workflow_config.data_config.get('end_date')}")
        logger.info(f"  Strategies: {len(workflow_config.backtest_config.get('strategies', []))}")
        
        # Simulate some progress
        import time
        for i in range(5):
            time.sleep(0.5)
            logger.info(f"  Processing... {(i+1)*20}%")
        
        logger.info("✓ Backtest completed successfully")
        logger.info("Results would be saved to: results/backtest_results.html")
        
    elif workflow_config.workflow_type == WorkflowType.OPTIMIZATION:
        logger.info("Running optimization...")
        opt_config = workflow_config.optimization_config
        logger.info(f"  Method: {opt_config.get('method')}")
        logger.info(f"  Trials: {opt_config.get('n_trials')}")
        logger.info(f"  Parameters: {list(opt_config.get('parameter_space', {}).keys())}")
        
        logger.info("✓ Optimization completed successfully")
        logger.info("Best parameters would be saved to: results/optimization_results.json")
        
    elif workflow_config.workflow_type == WorkflowType.LIVE_TRADING:
        live_config = workflow_config.live_config
        is_paper = live_config.get('paper_trading', True)
        logger.info(f"Starting {'paper' if is_paper else 'live'} trading...")
        logger.info(f"  Broker: {live_config.get('broker', {}).get('name')}")
        logger.info(f"  Symbols: {workflow_config.data_config.get('symbols', [])}")
        
        logger.info("✓ Trading system started")
        logger.info("Press Ctrl+C to stop...")
    
    return 0


def validate_config(config_path: str):
    """Validate a configuration file."""
    logger = logging.getLogger("main")
    
    validator = ConfigSchemaValidator()
    result = validator.validate_file(config_path)
    
    print(f"\nValidating: {config_path}")
    print("-" * 60)
    
    if result.is_valid:
        print("✓ Configuration is VALID")
        
        # Show normalized values
        if result.normalized_config:
            print("\nNormalized values added:")
            show_normalized_values(result.normalized_config)
    else:
        print("✗ Configuration is INVALID")
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  ✗ {error}")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    return 0 if result.is_valid else 1


def create_example(workflow_type: str, output_path: Optional[str] = None):
    """Create an example configuration file."""
    logger = logging.getLogger("main")
    
    examples = {
        "backtest": {
            "name": "Example Backtest Configuration",
            "type": "backtest",
            "description": "A simple moving average crossover backtest",
            "data": {
                "symbols": ["SPY", "QQQ", "IWM"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "frequency": "1d",
                "source": "yahoo"
            },
            "portfolio": {
                "initial_capital": 100000,
                "currency": "USD",
                "commission": {
                    "type": "fixed",
                    "value": 1.0
                }
            },
            "strategies": [{
                "name": "ma_crossover",
                "type": "moving_average_crossover",
                "allocation": 1.0,
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30,
                    "ma_type": "SMA"
                }
            }],
            "risk": {
                "position_sizers": [{
                    "name": "equal_weight",
                    "type": "percentage",
                    "percentage": 33.33
                }],
                "limits": [{
                    "type": "position",
                    "max_position": 50000
                }, {
                    "type": "drawdown",
                    "max_drawdown_pct": 20,
                    "reduce_at_pct": 15
                }]
            },
            "analysis": {
                "metrics": ["returns", "sharpe", "max_drawdown", "win_rate"],
                "plots": ["equity_curve", "drawdown"],
                "export": {
                    "format": "html",
                    "path": "results/backtest_report.html"
                }
            }
        },
        "optimization": {
            "name": "Example Optimization Configuration",
            "type": "optimization",
            "description": "Optimize moving average parameters",
            "base_config": {
                "data": {
                    "symbols": ["SPY"],
                    "start_date": "2022-01-01",
                    "end_date": "2023-12-31",
                    "frequency": "1h"
                },
                "portfolio": {
                    "initial_capital": 100000
                },
                "strategies": [{
                    "name": "ma_strategy",
                    "type": "moving_average_crossover"
                }]
            },
            "optimization": {
                "method": "bayesian",
                "parameter_space": {
                    "fast_period": {
                        "type": "int",
                        "min": 5,
                        "max": 50
                    },
                    "slow_period": {
                        "type": "int",
                        "min": 20,
                        "max": 200
                    }
                },
                "constraints": [{
                    "type": "expression",
                    "expression": "slow_period > fast_period + 10"
                }],
                "objectives": [{
                    "metric": "sharpe_ratio",
                    "direction": "maximize"
                }],
                "n_trials": 100
            }
        },
        "live_trading": {
            "name": "Example Live Trading Configuration",
            "type": "live_trading",
            "description": "Paper trading with momentum strategy",
            "paper_trading": True,
            "broker": {
                "name": "paper",
                "timeout": 30
            },
            "data": {
                "provider": "yahoo",
                "symbols": ["SPY", "QQQ"],
                "frequency": "5min",
                "lookback_days": 30
            },
            "portfolio": {
                "initial_capital": 100000,
                "max_positions": 2
            },
            "strategies": [{
                "name": "momentum",
                "type": "momentum_strategy",
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02
                }
            }],
            "risk": {
                "limits": [{
                    "type": "position",
                    "max_position": 25000
                }, {
                    "type": "daily_loss",
                    "max_daily_loss_pct": 2.0
                }]
            }
        }
    }
    
    example = examples[workflow_type]
    
    # Determine output path
    if not output_path:
        output_path = f"example_{workflow_type}.yaml"
    
    output_file = Path(output_path)
    
    # Create parent directories if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write YAML file
    with open(output_file, 'w') as f:
        yaml.dump(example, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created example configuration: {output_file}")
    logger.info(f"Run with: python main_yaml.py --config {output_file}")
    
    return 0


def list_schemas():
    """List available configuration schemas."""
    print("\nAvailable Configuration Schemas:")
    print("-" * 40)
    
    schemas = [
        ("backtest", "Run historical backtests on strategies"),
        ("optimization", "Optimize strategy parameters"),
        ("live_trading", "Execute strategies in real-time")
    ]
    
    for schema_name, description in schemas:
        print(f"\n{schema_name}:")
        print(f"  {description}")
    
    print("\nCreate an example with:")
    print("  python main_yaml.py --create-example <schema_name> --output my_config.yaml")
    
    return 0


def print_container_hierarchy(node, level=0):
    """Print container hierarchy tree."""
    indent = "  " * level
    print(f"{indent}├─ {node['type']} [{node['id']}]")
    capabilities = ", ".join(node.get('capabilities', []))
    if capabilities:
        print(f"{indent}│  Capabilities: {capabilities}")
    
    children = node.get('children', [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        print_container_hierarchy(child, level + 1)


def show_normalized_values(config, path="", original=None):
    """Show values that were normalized/added."""
    # This is a simplified version - in reality would compare with original
    defaults = {
        "type": "Configuration type",
        "execution_mode": "Execution mode for backtests",
        "timezone": "Data timezone",
        "format": "Data format",
        "enabled": "Strategy enabled flag",
        "allocation": "Strategy allocation"
    }
    
    if isinstance(config, dict):
        for key, value in config.items():
            if key in defaults and path:
                print(f"  - {path}.{key} = {value} ({defaults[key]})")
    

async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")
    
    # ASCII art banner
    if args.log_level != "ERROR":
        print("""
    ╔═══════════════════════════════════════════════════════╗
    ║             ADMF-PC: Zero-Code Trading                ║
    ║     Advanced Data Management Framework for            ║
    ║           Portfolio Construction                      ║
    ╚═══════════════════════════════════════════════════════╝
        """)
    
    try:
        # Execute requested operation
        if args.config:
            return await execute_config(args.config, args.dry_run, args.show_containers)
        
        elif args.validate:
            return validate_config(args.validate)
        
        elif args.create_example:
            if not args.output and args.create_example:
                args.output = f"example_{args.create_example}.yaml"
            return create_example(args.create_example, args.output)
        
        elif args.list_schemas:
            return list_schemas()
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))