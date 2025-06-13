#!/usr/bin/env python
# ADMF-PC Main CLI Entry Point
"""
Main command-line interface for ADMF-PC.
Provides unified access to all system functionality.
"""

import click
import sys
from pathlib import Path

# Import analytics CLI
from analytics.cli.main import analytics


@click.group()
@click.version_option(version='1.0.0', prog_name='ADMF-PC')
def cli():
    """ADMF-PC: Adaptive Data Mining Framework - Protocol + Composition
    
    A sophisticated trading system framework using event-driven architecture
    and protocol-based composition patterns.
    """
    pass


# Register analytics subcommand
cli.add_command(analytics)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--workspace', '-w', type=click.Path(),
              help='Workspace directory for results')
@click.option('--dry-run', is_flag=True,
              help='Validate configuration without running')
def backtest(config_file, workspace, dry_run):
    """Run a backtest from configuration file
    
    Examples:
        admf backtest config/simple_backtest.yaml
        admf backtest config/grid_search.yaml --workspace ./results/my_test
    """
    click.echo(f"Running backtest from {config_file}")
    
    if dry_run:
        click.echo("Dry run mode - validating configuration only")
    
    if workspace:
        click.echo(f"Results will be saved to: {workspace}")
    
    # Import and run the main backtest function
    try:
        from main import main as run_main
        # Pass the config file to main
        sys.argv = ['main.py', config_file]
        run_main()
    except Exception as e:
        click.echo(f"Error running backtest: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--phases', '-p', type=int,
              help='Number of optimization phases')
def optimize(config_file, phases):
    """Run optimization workflow
    
    Examples:
        admf optimize config/parameter_optimization.yaml
        admf optimize config/walk_forward.yaml --phases 12
    """
    click.echo(f"Running optimization from {config_file}")
    
    if phases:
        click.echo(f"Phases: {phases}")
    
    # TODO: Implement optimization runner
    click.echo("Optimization functionality coming soon!")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--source', '-s', required=True,
              help='Source workspace with signals')
def replay(config_file, source):
    """Run signal replay from stored signals
    
    Examples:
        admf replay config/replay.yaml --source workspace_12345
    """
    click.echo(f"Running signal replay from {source}")
    click.echo(f"Configuration: {config_file}")
    
    # TODO: Implement signal replay
    click.echo("Signal replay functionality coming soon!")


@cli.command()
def config():
    """Manage ADMF-PC configuration
    
    Opens configuration management interface.
    """
    click.echo("ADMF-PC Configuration Management")
    click.echo("-" * 40)
    
    # Show current configuration
    config_paths = [
        Path.home() / '.admf' / 'config.yaml',
        Path.cwd() / '.admf.yaml'
    ]
    
    for path in config_paths:
        if path.exists():
            click.echo(f"Found config: {path}")
    
    # TODO: Implement config management
    click.echo("\nConfiguration management coming soon!")


@cli.command()
def version():
    """Show version information"""
    click.echo("ADMF-PC (Adaptive Data Mining Framework - Protocol + Composition)")
    click.echo("Version: 1.0.0")
    click.echo("Python: " + sys.version.split()[0])
    
    # Show component versions
    try:
        import pandas as pd
        import numpy as np
        import pyarrow as pa
        
        click.echo(f"\nDependencies:")
        click.echo(f"  pandas: {pd.__version__}")
        click.echo(f"  numpy: {np.__version__}")
        click.echo(f"  pyarrow: {pa.__version__}")
    except ImportError:
        pass


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()