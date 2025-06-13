# Scan Command - Discovery & Overview
"""
Scan workspaces for optimization runs and provide quick overviews.
"""

import click
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
import pandas as pd

from ...storage.metadata_scanner import GridSearchScanner


@click.command()
@click.option('--type', 'workflow_type', 
              type=click.Choice(['grid_search', 'optimization', 'backtest', 'all']),
              default='all',
              help='Filter by workflow type')
@click.option('--after', type=click.DateTime(),
              help='Show runs after this date')
@click.option('--symbols', multiple=True,
              help='Filter by symbols')
@click.option('--min-sharpe', type=float,
              help='Minimum Sharpe ratio')
@click.option('--last', type=int,
              help='Show only last N runs')
@click.option('--detailed', '-d', is_flag=True,
              help='Show detailed information')
@click.pass_context
def scan(ctx, workflow_type, after, symbols, min_sharpe, last, detailed):
    """Scan workspaces for optimization runs
    
    Examples:
        admf analytics scan --last 10
        admf analytics scan --type grid_search --min-sharpe 1.5
        admf analytics scan --symbols SPY QQQ --after 2025-01-01
    """
    workspace_root = ctx.obj['workspace_root']
    scanner = GridSearchScanner(workspace_root)
    
    # Scan workspaces
    if workflow_type == 'all':
        workflow_type = None
    
    df = scanner.scan_all_workspaces(
        workflow_type=workflow_type,
        after_date=after,
        symbols=list(symbols) if symbols else None
    )
    
    if df.empty:
        click.echo("No workspaces found matching criteria")
        return
    
    # Apply performance filter if specified
    if min_sharpe is not None:
        df = df[df['best_sharpe'] >= min_sharpe]
    
    # Sort by date
    df = df.sort_values('created_at', ascending=False)
    
    # Limit results if requested
    if last:
        df = df.head(last)
    
    # Format for display
    if detailed:
        show_detailed_scan(df)
    else:
        show_summary_scan(df)
    
    # Show totals
    click.echo(f"\nTotal runs found: {len(df)}")
    
    if 'best_sharpe' in df.columns and df['best_sharpe'].notna().any():
        click.echo(f"Average best Sharpe: {df['best_sharpe'].mean():.2f}")
        click.echo(f"Top Sharpe ratio: {df['best_sharpe'].max():.2f}")


def show_summary_scan(df: pd.DataFrame):
    """Show summary scan results"""
    # Prepare data for tabulate
    rows = []
    for _, row in df.iterrows():
        created = datetime.fromisoformat(row['created_at']).strftime('%Y-%m-%d %H:%M')
        
        # Format workspace name (truncate if too long)
        workspace = row['workspace']
        if len(workspace) > 40:
            workspace = workspace[:37] + "..."
        
        # Format performance
        sharpe = f"{row['best_sharpe']:.2f}" if pd.notna(row.get('best_sharpe')) else "N/A"
        
        rows.append([
            workspace,
            created,
            row['workflow_type'],
            row['symbols'],
            row['total_strategies'],
            sharpe,
            row.get('best_strategy', 'N/A')[:20]
        ])
    
    headers = ['Workspace', 'Created', 'Type', 'Symbols', 'Strategies', 'Best Sharpe', 'Best Strategy']
    
    click.echo("\n" + tabulate(rows, headers=headers, tablefmt='simple'))


def show_detailed_scan(df: pd.DataFrame):
    """Show detailed scan results"""
    for idx, row in df.iterrows():
        click.echo("\n" + "="*80)
        click.echo(f"Workspace: {row['workspace']}")
        click.echo(f"Path: {row['path']}")
        click.echo("-"*40)
        
        created = datetime.fromisoformat(row['created_at'])
        click.echo(f"Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"Type: {row['workflow_type']}")
        click.echo(f"Symbols: {row['symbols']}")
        
        click.echo(f"\nStrategies: {row['total_strategies']}")
        click.echo(f"Classifiers: {row.get('total_classifiers', 0)}")
        
        if pd.notna(row.get('best_sharpe')):
            click.echo(f"\nBest Performance:")
            click.echo(f"  Strategy: {row.get('best_strategy', 'N/A')}")
            click.echo(f"  Sharpe Ratio: {row['best_sharpe']:.3f}")
        
        # Could add more details from manifest if available
        
    click.echo("\n" + "="*80)


@click.command('scan-performance')
@click.argument('workspace', required=False)
@click.option('--min-sharpe', type=float,
              help='Minimum Sharpe ratio')
@click.option('--max-drawdown', type=float,
              help='Maximum drawdown (positive number)')
@click.option('--strategy-types', multiple=True,
              help='Filter by strategy types')
@click.option('--top', type=int, default=20,
              help='Show top N strategies')
@click.option('--export', type=click.Path(),
              help='Export results to CSV')
@click.pass_context
def scan_performance(ctx, workspace, min_sharpe, max_drawdown, strategy_types, top, export):
    """Scan strategy performance across workspaces
    
    Examples:
        admf analytics scan-performance --min-sharpe 1.5 --top 10
        admf analytics scan-performance my_workspace --export results.csv
    """
    workspace_root = ctx.obj['workspace_root']
    scanner = GridSearchScanner(workspace_root)
    
    # Scan performance
    df = scanner.scan_performance(
        workspace_name=workspace,
        min_sharpe=min_sharpe,
        max_drawdown=max_drawdown,
        strategy_types=list(strategy_types) if strategy_types else None
    )
    
    if df.empty:
        click.echo("No strategies found matching criteria")
        return
    
    # Limit to top N
    if top and len(df) > top:
        df = df.head(top)
    
    # Display results
    display_performance_results(df)
    
    # Export if requested
    if export:
        df.to_csv(export, index=False)
        click.echo(f"\nResults exported to {export}")


def display_performance_results(df: pd.DataFrame):
    """Display performance scan results"""
    # Group by strategy type
    for strategy_type in df['strategy_type'].unique():
        type_df = df[df['strategy_type'] == strategy_type]
        
        click.echo(f"\n{strategy_type.upper()} Strategies ({len(type_df)} variants)")
        click.echo("-" * 80)
        
        # Prepare rows
        rows = []
        for _, row in type_df.iterrows():
            # Format parameters
            params_str = format_params(row['params'])
            
            rows.append([
                row['strategy_id'][:30],
                f"{row['sharpe']:.3f}" if pd.notna(row['sharpe']) else "N/A",
                f"{row['max_drawdown']:.1%}" if pd.notna(row['max_drawdown']) else "N/A",
                f"{row['win_rate']:.1%}" if pd.notna(row['win_rate']) else "N/A",
                row['signal_changes'],
                f"{row['compression_ratio']:.3f}",
                params_str[:40]
            ])
        
        headers = ['Strategy ID', 'Sharpe', 'Max DD', 'Win Rate', 'Signals', 'Compression', 'Parameters']
        click.echo(tabulate(rows, headers=headers, tablefmt='simple'))


def format_params(params: dict) -> str:
    """Format parameters dictionary for display"""
    if not params:
        return ""
    
    # Show key parameters
    parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    
    return ", ".join(parts)