# Migrate Command - Workspace Migration
"""
Migrate workspaces from old UUID format to new standardized format.
"""

import click
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

from ...storage.workspace_migrator import WorkspaceMigrator


@click.command()
@click.option('--source', '-s', 
              type=click.Path(exists=True),
              default='./workspaces',
              help='Source directory with old workspaces')
@click.option('--destination', '-d',
              type=click.Path(),
              help='Destination directory for migrated workspaces')
@click.option('--dry-run', is_flag=True,
              help='Show what would be migrated without doing it')
@click.option('--workspace', '-w',
              help='Migrate specific workspace by ID')
@click.option('--force', '-f', is_flag=True,
              help='Force migration even if destination exists')
@click.pass_context
def migrate(ctx, source, destination, dry_run, workspace, force):
    """Migrate workspaces to new format
    
    Converts old UUID-based workspace format to new standardized format
    with improved directory naming and sparse Parquet storage.
    
    Examples:
        admf analytics migrate --dry-run
        admf analytics migrate --workspace fc4bb91c-2cea-441b-85e4-10d83a0e1580
        admf analytics migrate --source ./old_workspaces --destination ./new_workspaces
    """
    source_path = Path(source)
    
    # Default destination is source_new
    if destination:
        dest_path = Path(destination)
    else:
        dest_path = source_path.parent / f"{source_path.name}_migrated"
    
    # Create migrator
    migrator = WorkspaceMigrator(source_path, dest_path)
    
    if workspace:
        # Migrate specific workspace
        if dry_run:
            click.echo(f"Would migrate workspace: {workspace}")
            old_path = source_path / workspace
            new_path = migrator._plan_migration(old_path)
            click.echo(f"  From: {old_path}")
            click.echo(f"  To: {new_path}")
        else:
            try:
                new_path = migrator.migrate_workspace(workspace)
                click.echo(f"Successfully migrated {workspace}")
                click.echo(f"  New location: {new_path}")
            except Exception as e:
                click.echo(f"Error migrating {workspace}: {e}", err=True)
    else:
        # Migrate all workspaces
        click.echo(f"Source: {source_path}")
        click.echo(f"Destination: {dest_path}")
        
        if not dry_run and not dest_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of migrations
        migrations = migrator.migrate_all(dry_run=True)
        
        if not migrations:
            click.echo("\nNo workspaces found to migrate")
            return
        
        # Display migration plan
        click.echo(f"\nFound {len(migrations)} workspaces to migrate:")
        click.echo("-" * 80)
        
        rows = []
        for old, new in migrations:
            old_name = Path(old).name
            new_name = Path(new).name
            
            # Truncate if too long
            if len(old_name) > 40:
                old_name = old_name[:37] + "..."
            if len(new_name) > 60:
                new_name = new_name[:57] + "..."
            
            rows.append([old_name, "→", new_name])
        
        click.echo(tabulate(rows, tablefmt='plain'))
        
        if dry_run:
            click.echo("\nThis is a dry run. No files will be modified.")
            click.echo("Remove --dry-run flag to perform migration.")
        else:
            # Confirm migration
            if not force:
                click.confirm("\nProceed with migration?", abort=True)
            
            # Perform migration
            click.echo("\nMigrating workspaces...")
            
            success_count = 0
            error_count = 0
            
            with click.progressbar(migrations, label='Migrating') as migration_list:
                for old_path, _ in migration_list:
                    try:
                        workspace_id = Path(old_path).name
                        migrator.migrate_workspace(workspace_id)
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        click.echo(f"\nError migrating {workspace_id}: {e}", err=True)
            
            # Summary
            click.echo(f"\nMigration complete:")
            click.echo(f"  Successful: {success_count}")
            click.echo(f"  Failed: {error_count}")
            
            if success_count > 0:
                click.echo(f"\nMigrated workspaces are in: {dest_path}")


@click.command('migrate-signals')
@click.argument('signal_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output Parquet file')
@click.option('--show-stats', is_flag=True,
              help='Show compression statistics')
@click.pass_context
def migrate_signals(ctx, signal_file, output, show_stats):
    """Convert JSON signal file to Parquet format
    
    Converts sparse JSON signal storage to efficient Parquet format.
    
    Example:
        admf analytics migrate-signals signals_strategy_SPY_ma_crossover.json -o signals.parquet
    """
    from ...storage.sparse_storage import SparseSignalStorage
    import json
    
    signal_path = Path(signal_file)
    
    # Load JSON data
    with open(signal_path) as f:
        data = json.load(f)
    
    # Extract metadata and changes
    metadata = data.get('metadata', {})
    changes = data.get('changes', [])
    total_bars = metadata.get('total_bars', 0)
    
    # Convert to DataFrame
    df = SparseSignalStorage.from_json_changes(changes, total_bars, metadata)
    
    # Determine output path
    if not output:
        output = signal_path.with_suffix('.parquet')
    
    # Save as Parquet
    SparseSignalStorage.to_parquet(df, Path(output))
    
    click.echo(f"Converted {signal_path.name} to Parquet format")
    click.echo(f"Output: {output}")
    
    if show_stats:
        click.echo("\nCompression Statistics:")
        click.echo(f"  Total bars: {total_bars:,}")
        click.echo(f"  Signal changes: {len(changes):,}")
        click.echo(f"  Compression ratio: {df.attrs['compression_ratio']:.4f}")
        click.echo(f"  Space savings: {(1 - df.attrs['compression_ratio']) * 100:.1f}%")
        
        # Check file sizes
        json_size = signal_path.stat().st_size
        parquet_size = Path(output).stat().st_size
        
        click.echo(f"\nFile sizes:")
        click.echo(f"  JSON: {json_size:,} bytes")
        click.echo(f"  Parquet: {parquet_size:,} bytes")
        click.echo(f"  Reduction: {(1 - parquet_size/json_size) * 100:.1f}%")