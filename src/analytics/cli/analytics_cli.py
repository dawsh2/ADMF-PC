#!/usr/bin/env python3
"""
Analytics CLI - Main entry point for all analytics operations.
"""
import click
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.analytics.strategy_filter import StrategyFilter, analyze_grid_search
from src.analytics.mining.pattern_miner import PatternMiner


@click.group()
def cli():
    """ADMF-PC Analytics Toolkit"""
    pass


@cli.command()
@click.argument('workspace', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--min-trades', default=10, help='Minimum trades to consider viable')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe ratio')
@click.option('--max-correlation', default=0.7, help='Maximum correlation between strategies')
@click.option('--commission', default=0.01, help='Commission percentage')
@click.option('--slippage', default=0.01, help='Slippage percentage')
@click.option('--classifier', type=click.Path(exists=True), help='Path to classifier signals')
def filter(workspace, data_path, min_trades, min_sharpe, max_correlation, commission, slippage, classifier):
    """Filter strategies based on performance criteria."""
    click.echo(f"Analyzing workspace: {workspace}")
    
    results = analyze_grid_search(
        workspace_path=workspace,
        data_path=data_path,
        classifier_path=classifier,
        min_trades=min_trades,
        min_sharpe=min_sharpe,
        max_correlation=max_correlation,
        commission_pct=commission,
        slippage_pct=slippage
    )
    
    # Print summary
    click.echo("\n=== Analysis Summary ===")
    for name, df in results.items():
        click.echo(f"{name}: {len(df)} entries")


@cli.command()
@click.argument('workspace', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--max-lag', default=10, help='Maximum lag for signal combinations')
@click.option('--output', type=click.Path(), help='Output directory for results')
def mine(workspace, data_path, max_lag, output):
    """Mine for emergent patterns and signal combinations."""
    click.echo(f"Mining patterns in workspace: {workspace}")
    
    miner = PatternMiner(workspace)
    try:
        output_dir = Path(output) if output else None
        results = miner.mine_all_patterns(data_path, output_dir)
        
        # Show top insights
        click.echo("\n=== Top Insights ===")
        
        if not results['signal_combinations'].empty:
            click.echo("\nBest Signal Combinations:")
            top_combos = results['signal_combinations'].nlargest(3, 'avg_return')
            for _, row in top_combos.iterrows():
                click.echo(f"  Enter: {row['entry_type']}, Exit: {row['exit_type']} "
                          f"(signal={row['exit_signal']}), "
                          f"Return: {row['avg_return']:.2f}%")
        
        if not results['exit_strategies'].empty:
            click.echo("\nOptimal Exit Strategies:")
            for _, row in results['exit_strategies'].iterrows():
                click.echo(f"  {row['strategy_type']}: "
                          f"1bar={row['avg_return_1bar']:.2f}%, "
                          f"5bar={row['avg_return_5bar']:.2f}%, "
                          f"10bar={row['avg_return_10bar']:.2f}%")
    finally:
        miner.close()


@cli.command()
@click.argument('workspace', type=click.Path(exists=True))
@click.argument('strategy_name')
@click.argument('data_path', type=click.Path(exists=True))
def sensitivity(workspace, strategy_name, data_path):
    """Analyze parameter sensitivity for a strategy type."""
    click.echo(f"Analyzing parameter sensitivity for {strategy_name}")
    
    filter = StrategyFilter(workspace)
    try:
        results = filter.sensitivity_analysis(strategy_name, data_path)
        
        if results.empty:
            click.echo("No stable parameter neighborhoods found")
        else:
            click.echo(f"\nFound {len(results)} stable parameter combinations:")
            click.echo(results.to_string(index=False))
    finally:
        filter.close()


@cli.command()
@click.argument('workspace', type=click.Path(exists=True))
@click.option('--query', '-q', help='SQL query to execute')
@click.option('--file', '-f', type=click.Path(exists=True), help='SQL file to execute')
def query(workspace, query, file):
    """Execute SQL queries on the analytics database."""
    import duckdb
    
    db_path = Path(workspace) / "analytics.duckdb"
    con = duckdb.connect(str(db_path))
    
    try:
        if file:
            with open(file, 'r') as f:
                sql = f.read()
        elif query:
            sql = query
        else:
            click.echo("Provide either --query or --file")
            return
        
        result = con.execute(sql).df()
        click.echo(result.to_string())
    finally:
        con.close()


@cli.command()
@click.argument('workspace', type=click.Path(exists=True))
def summary(workspace):
    """Show workspace summary and available data."""
    from datetime import datetime
    import json
    
    workspace_path = Path(workspace)
    
    # Check for metadata
    metadata_file = workspace_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        click.echo(f"Workspace: {metadata.get('id', 'Unknown')}")
        click.echo(f"Created: {metadata.get('timestamp', 'Unknown')}")
        click.echo(f"Config: {metadata.get('config_path', 'Unknown')}")
    
    # Count signals
    signal_files = list(workspace_path.glob("traces/*/signals/*/*.parquet"))
    classifier_files = list(workspace_path.glob("traces/*/classifiers/*/*.parquet"))
    
    click.echo(f"\nSignal files: {len(signal_files)}")
    click.echo(f"Classifier files: {len(classifier_files)}")
    
    # Get unique strategies
    strategies = set()
    for f in signal_files:
        strategies.add(f.parent.name)
    
    click.echo(f"\nStrategy types: {', '.join(sorted(strategies))}")
    
    # Check analytics DB
    db_path = workspace_path / "analytics.duckdb"
    if db_path.exists():
        import duckdb
        con = duckdb.connect(str(db_path))
        try:
            # Count total signals
            result = con.execute("""
                SELECT COUNT(*) as total_signals
                FROM read_parquet('traces/*/signals/*/*.parquet')
                WHERE val != 0
            """).fetchone()
            click.echo(f"\nTotal signals: {result[0]}")
        except:
            pass
        finally:
            con.close()


if __name__ == '__main__':
    cli()