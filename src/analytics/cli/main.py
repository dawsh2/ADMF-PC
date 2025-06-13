# Analytics CLI Main Entry Point
"""
Main CLI interface for ADMF-PC analytics operations.
"""

import click
from pathlib import Path
from datetime import datetime
import json
import sys
from typing import Optional

from ..storage.metadata_scanner import GridSearchScanner
from ..storage.workspace_migrator import WorkspaceMigrator
from .commands import scan, analyze, migrate, report


@click.group()
@click.option('--workspace-root', '-w', 
              default='./workspaces',
              type=click.Path(exists=True),
              help='Root directory for workspaces')
@click.pass_context
def analytics(ctx, workspace_root):
    """ADMF-PC Analytics - Analyze optimization and backtest results"""
    ctx.ensure_object(dict)
    ctx.obj['workspace_root'] = Path(workspace_root)


# Register commands
analytics.add_command(scan.scan)
analytics.add_command(analyze.analyze)
analytics.add_command(migrate.migrate)
analytics.add_command(report.report)


@analytics.command()
@click.pass_context
def interactive(ctx):
    """Enter interactive analytics mode"""
    click.echo("ADMF Analytics Interactive Mode")
    click.echo("Type 'help' for available commands or 'exit' to quit")
    
    workspace_root = ctx.obj['workspace_root']
    scanner = GridSearchScanner(workspace_root)
    current_workspace = None
    
    while True:
        # Build prompt
        if current_workspace:
            prompt = f"ADMF Analytics [{current_workspace}]> "
        else:
            prompt = "ADMF Analytics> "
        
        try:
            command = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nExiting...")
            break
        
        if not command:
            continue
        
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd in ['exit', 'quit', 'q']:
            break
        elif cmd == 'help':
            show_interactive_help()
        elif cmd == 'scan':
            handle_scan(scanner, parts[1:])
        elif cmd == 'select':
            if len(parts) > 1:
                current_workspace = parts[1]
                click.echo(f"Selected workspace: {current_workspace}")
            else:
                click.echo("Usage: select <workspace_name>")
        elif cmd == 'clear':
            current_workspace = None
            click.echo("Cleared workspace selection")
        elif cmd == 'strategies':
            if current_workspace:
                show_strategies(scanner, current_workspace)
            else:
                click.echo("No workspace selected. Use 'select <workspace_name>' first")
        else:
            click.echo(f"Unknown command: {cmd}")


def show_interactive_help():
    """Show help for interactive mode"""
    help_text = """
Available commands:
  scan [--last N]          Show workspaces (optionally last N)
  select <workspace>       Select a workspace for analysis
  clear                    Clear workspace selection
  strategies               List strategies in selected workspace
  classifiers              List classifiers in selected workspace
  performance              Show performance summary
  correlations             Show strategy correlations
  help                     Show this help
  exit/quit/q             Exit interactive mode
"""
    click.echo(help_text)


def handle_scan(scanner, args):
    """Handle scan command in interactive mode"""
    limit = None
    
    # Parse arguments
    if '--last' in args:
        try:
            idx = args.index('--last')
            if idx + 1 < len(args):
                limit = int(args[idx + 1])
        except (ValueError, IndexError):
            click.echo("Invalid --last argument")
            return
    
    # Perform scan
    df = scanner.scan_all_workspaces()
    
    if df.empty:
        click.echo("No workspaces found")
        return
    
    # Sort by date
    df = df.sort_values('created_at', ascending=False)
    
    if limit:
        df = df.head(limit)
    
    # Display results
    click.echo("\nFound workspaces:")
    click.echo("-" * 80)
    
    for _, row in df.iterrows():
        created = datetime.fromisoformat(row['created_at']).strftime('%Y-%m-%d %H:%M')
        click.echo(
            f"{row['workspace']:<40} | "
            f"{created} | "
            f"{row['total_strategies']:>4} strategies | "
            f"Best Sharpe: {row.get('best_sharpe', 'N/A'):>6}"
        )


def show_strategies(scanner, workspace_name):
    """Show strategies in a workspace"""
    df = scanner.scan_performance(workspace_name)
    
    if df.empty:
        click.echo("No strategies found")
        return
    
    # Group by type
    for strategy_type in df['strategy_type'].unique():
        type_df = df[df['strategy_type'] == strategy_type]
        click.echo(f"\n{strategy_type} ({len(type_df)} variants):")
        click.echo("-" * 60)
        
        # Show top 5 by Sharpe
        top_5 = type_df.nlargest(5, 'sharpe', keep='first')
        
        for _, row in top_5.iterrows():
            params_str = json.dumps(row['params'], separators=(',', ':'))
            if len(params_str) > 30:
                params_str = params_str[:27] + "..."
            
            click.echo(
                f"  {row['strategy_id']:<30} | "
                f"Sharpe: {row['sharpe']:>6.2f} | "
                f"DD: {row['max_drawdown']:>6.1%} | "
                f"{params_str}"
            )


# Make CLI available as main entry point
def main():
    """Main entry point for the analytics CLI"""
    analytics(obj={})


if __name__ == '__main__':
    main()