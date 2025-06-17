#!/usr/bin/env python3
"""
Utility to optimize existing metadata.json files by removing redundant data.

This script can be run on existing workspaces to reduce their metadata.json file sizes.
"""

import json
import shutil
from pathlib import Path
import argparse


def optimize_metadata_file(metadata_path: Path, backup: bool = True) -> dict:
    """
    Optimize a metadata.json file by removing redundant data.
    
    Args:
        metadata_path: Path to the metadata.json file
        backup: Whether to create a backup before modifying
        
    Returns:
        Dict with optimization results
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    original_size = len(json.dumps(metadata, indent=2))
    
    # Create backup if requested
    if backup:
        backup_path = metadata_path.with_suffix('.json.backup')
        shutil.copy2(metadata_path, backup_path)
        print(f"📁 Created backup: {backup_path}")
    
    # Optimize strategy metadata
    bytes_saved = 0
    optimizations_applied = []
    
    if 'strategy_metadata' in metadata:
        strategy_meta = metadata['strategy_metadata']
        
        # Remove default_regime_strategies duplication
        strategies = strategy_meta.get('strategies', {})
        for strategy_name, strategy_data in strategies.items():
            if 'default_regime_strategies' in strategy_data:
                removed_size = len(json.dumps(strategy_data['default_regime_strategies']))
                del strategy_data['default_regime_strategies']
                bytes_saved += removed_size
                optimizations_applied.append(f"Removed default_regime_strategies from {strategy_name}")
        
        # Remove component_inventory (can be computed on-demand)
        if 'component_inventory' in strategy_meta:
            removed_size = len(json.dumps(strategy_meta['component_inventory']))
            del strategy_meta['component_inventory']
            bytes_saved += removed_size
            optimizations_applied.append("Removed component_inventory (computable on-demand)")
    
    # Save optimized metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    optimized_size = len(json.dumps(metadata, indent=2))
    savings_pct = (bytes_saved / original_size) * 100
    
    return {
        'original_size': original_size,
        'optimized_size': optimized_size,
        'bytes_saved': bytes_saved,
        'savings_percentage': savings_pct,
        'optimizations_applied': optimizations_applied
    }


def find_metadata_files(workspace_dir: Path) -> list:
    """Find all metadata.json files in workspace directories."""
    metadata_files = []
    
    if workspace_dir.is_dir():
        # Look for metadata.json files in subdirectories
        for subdir in workspace_dir.iterdir():
            if subdir.is_dir():
                metadata_file = subdir / 'metadata.json'
                if metadata_file.exists():
                    metadata_files.append(metadata_file)
    
    return metadata_files


def main():
    parser = argparse.ArgumentParser(description='Optimize metadata.json files by removing redundant data')
    parser.add_argument('path', help='Path to metadata.json file or workspace directory')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be optimized without making changes')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    backup = not args.no_backup
    
    print("🔧 METADATA OPTIMIZATION UTILITY")
    print("=" * 50)
    
    # Find metadata files to process
    if path.is_file() and path.name == 'metadata.json':
        metadata_files = [path]
    elif path.is_dir():
        metadata_files = find_metadata_files(path)
        if not metadata_files:
            print(f"❌ No metadata.json files found in {path}")
            return
    else:
        print(f"❌ Invalid path: {path}")
        return
    
    print(f"📁 Found {len(metadata_files)} metadata.json file(s) to process")
    
    total_original = 0
    total_optimized = 0
    total_saved = 0
    
    for metadata_file in metadata_files:
        print(f"\n🔍 Processing: {metadata_file}")
        
        try:
            if args.dry_run:
                # Load and analyze without modifying
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                original_size = len(json.dumps(metadata, indent=2))
                
                # Calculate potential savings
                potential_savings = 0
                if 'strategy_metadata' in metadata:
                    strategy_meta = metadata['strategy_metadata']
                    
                    # Check default_regime_strategies
                    strategies = strategy_meta.get('strategies', {})
                    for strategy_data in strategies.values():
                        if 'default_regime_strategies' in strategy_data:
                            potential_savings += len(json.dumps(strategy_data['default_regime_strategies']))
                    
                    # Check component_inventory
                    if 'component_inventory' in strategy_meta:
                        potential_savings += len(json.dumps(strategy_meta['component_inventory']))
                
                savings_pct = (potential_savings / original_size) * 100 if original_size > 0 else 0
                
                print(f"  📊 Original size: {original_size:,} bytes")
                print(f"  💡 Potential savings: {potential_savings:,} bytes ({savings_pct:.1f}%)")
                
            else:
                # Actually optimize the file
                result = optimize_metadata_file(metadata_file, backup=backup)
                
                print(f"  📊 Original size: {result['original_size']:,} bytes")
                print(f"  📉 Optimized size: {result['optimized_size']:,} bytes")
                print(f"  💾 Saved: {result['bytes_saved']:,} bytes ({result['savings_percentage']:.1f}%)")
                
                if result['optimizations_applied']:
                    print("  ✅ Optimizations applied:")
                    for opt in result['optimizations_applied']:
                        print(f"    - {opt}")
                
                total_original += result['original_size']
                total_optimized += result['optimized_size']
                total_saved += result['bytes_saved']
        
        except Exception as e:
            print(f"  ❌ Error processing {metadata_file}: {e}")
    
    if not args.dry_run and len(metadata_files) > 1:
        total_savings_pct = (total_saved / total_original) * 100 if total_original > 0 else 0
        print(f"\n📈 TOTAL RESULTS:")
        print(f"  Original total: {total_original:,} bytes")
        print(f"  Optimized total: {total_optimized:,} bytes")
        print(f"  Total saved: {total_saved:,} bytes ({total_savings_pct:.1f}%)")
    
    print("\n✅ Optimization complete!")
    if args.dry_run:
        print("💡 Run without --dry-run to apply optimizations")


if __name__ == "__main__":
    main()