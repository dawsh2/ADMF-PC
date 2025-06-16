#!/usr/bin/env python3
"""
Verify workspace structure creation for ADMF-PC.

This script checks:
1. What MultiStrategyTracer creates in the workspace
2. Where metadata.json is created
3. If analytics.db is being created by analytics integration
4. Complete workspace structure matches requirements
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List


def check_workspace_structure(workspace_path: Path) -> Dict[str, Any]:
    """Check what files and directories exist in a workspace."""
    result = {
        'workspace_path': str(workspace_path),
        'exists': workspace_path.exists(),
        'files': {},
        'directories': {},
        'structure_valid': False
    }
    
    if not workspace_path.exists():
        return result
    
    # Check for expected files
    expected_files = {
        'metadata.json': workspace_path / 'metadata.json',
        'analytics.db': workspace_path / 'analytics.duckdb',  # Note: it's .duckdb not .db
        'analytics.duckdb': workspace_path / 'analytics.duckdb'
    }
    
    for name, path in expected_files.items():
        result['files'][name] = {
            'exists': path.exists(),
            'path': str(path),
            'size': path.stat().st_size if path.exists() else 0
        }
    
    # Check for expected directories
    expected_dirs = {
        'traces': workspace_path / 'traces',
        'signals': workspace_path / 'traces' / 'signals',
        'classifiers': workspace_path / 'traces' / 'classifiers'
    }
    
    for name, path in expected_dirs.items():
        result['directories'][name] = {
            'exists': path.exists(),
            'path': str(path),
            'contents': []
        }
        
        if path.exists():
            # List subdirectories
            try:
                subdirs = [d.name for d in path.iterdir() if d.is_dir()]
                result['directories'][name]['contents'] = subdirs[:10]  # First 10
            except:
                pass
    
    # Check if structure is valid
    result['structure_valid'] = (
        result['files']['metadata.json']['exists'] and
        result['files']['analytics.duckdb']['exists'] and
        result['directories']['traces']['exists']
    )
    
    return result


def find_recent_workspaces(base_path: Path = Path('.'), limit: int = 5) -> List[Path]:
    """Find recent workspace directories."""
    workspaces = []
    
    # Common workspace patterns
    patterns = [
        'workspaces/expansive_grid_search_*',
        'workspaces/20*',
        'expansive_grid_search_*',
        '20*_*_*'  # Date-based pattern
    ]
    
    for pattern in patterns:
        workspaces.extend(base_path.glob(pattern))
    
    # Sort by modification time
    workspaces = sorted(set(workspaces), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    
    return workspaces[:limit]


def check_metadata_content(metadata_path: Path) -> Dict[str, Any]:
    """Check metadata.json content."""
    if not metadata_path.exists():
        return {'exists': False}
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return {
            'exists': True,
            'workspace_path': data.get('workspace_path'),
            'total_bars': data.get('total_bars'),
            'total_signals': data.get('total_signals'),
            'total_classifications': data.get('total_classifications'),
            'components_count': len(data.get('components', {})),
            'component_types': list(set(
                comp.get('component_type', 'unknown') 
                for comp in data.get('components', {}).values()
            ))
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def main():
    """Main verification function."""
    print("=== ADMF-PC Workspace Structure Verification ===\n")
    
    # Find recent workspaces
    base_path = Path('/Users/daws/ADMF-PC')
    workspaces = find_recent_workspaces(base_path)
    
    if not workspaces:
        print("No workspace directories found!")
        print("\nSearching for any directory with workspace-like structure...")
        
        # Try to find any directory with metadata.json
        metadata_files = list(base_path.glob('**/metadata.json'))
        if metadata_files:
            print(f"\nFound {len(metadata_files)} metadata.json files:")
            for mf in metadata_files[:5]:
                print(f"  - {mf.parent}")
                workspaces.append(mf.parent)
    
    # Check each workspace
    for i, workspace in enumerate(workspaces):
        print(f"\n{'='*60}")
        print(f"Workspace {i+1}: {workspace.name}")
        print(f"Path: {workspace}")
        print(f"{'='*60}")
        
        # Check structure
        structure = check_workspace_structure(workspace)
        
        print("\nFiles:")
        for name, info in structure['files'].items():
            status = "✓" if info['exists'] else "✗"
            print(f"  {status} {name}: {info['exists']}")
            if info['exists'] and info['size'] > 0:
                print(f"    Size: {info['size']:,} bytes")
        
        print("\nDirectories:")
        for name, info in structure['directories'].items():
            status = "✓" if info['exists'] else "✗"
            print(f"  {status} {name}: {info['exists']}")
            if info['contents']:
                print(f"    Contents: {', '.join(info['contents'][:5])}")
                if len(info['contents']) > 5:
                    print(f"    ... and {len(info['contents']) - 5} more")
        
        print(f"\nStructure Valid: {'YES' if structure['structure_valid'] else 'NO'}")
        
        # Check metadata content
        metadata_path = workspace / 'metadata.json'
        if metadata_path.exists():
            print("\nMetadata Content:")
            metadata = check_metadata_content(metadata_path)
            if 'error' in metadata:
                print(f"  Error reading metadata: {metadata['error']}")
            else:
                print(f"  Total bars: {metadata.get('total_bars', 'N/A')}")
                print(f"  Total signals: {metadata.get('total_signals', 'N/A')}")
                print(f"  Total classifications: {metadata.get('total_classifications', 'N/A')}")
                print(f"  Components: {metadata.get('components_count', 0)}")
                print(f"  Component types: {', '.join(metadata.get('component_types', []))}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("\nKey findings:")
    print("1. MultiStrategyTracer creates:")
    print("   - metadata.json at workspace root (line 303-306 in multi_strategy_tracer.py)")
    print("   - traces/SYMBOL_TIMEFRAME/{signals,classifiers}/STRATEGY_TYPE/ directories")
    print("   - Parquet files for each strategy/classifier")
    print("\n2. Analytics integration creates:")
    print("   - analytics.duckdb (not analytics.db) via setup_workspace() call")
    print("   - Created in workspace root by AnalyticsWorkspace initialization")
    print("\n3. Expected complete structure:")
    print("   workspace/")
    print("   ├── metadata.json          (created by MultiStrategyTracer)")
    print("   ├── analytics.duckdb       (created by AnalyticsIntegrator)")
    print("   └── traces/")
    print("       └── SYMBOL_TIMEFRAME/")
    print("           ├── signals/")
    print("           │   └── STRATEGY_TYPE/")
    print("           │       └── *.parquet")
    print("           └── classifiers/")
    print("               └── CLASSIFIER_TYPE/")
    print("                   └── *.parquet")


if __name__ == "__main__":
    main()