#!/usr/bin/env python3
"""
Migrate existing JSON signal files to hierarchical Parquet storage.

This script:
1. Finds all JSON signal files in workspaces/tmp/
2. Converts them to Parquet format
3. Organizes them in the hierarchical structure
4. Creates index files for each strategy/classifier type
"""

from pathlib import Path
import json
import logging
from typing import List, Dict, Any

from src.analytics.storage.hierarchical_parquet_storage import HierarchicalParquetStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_signal_json_files(base_path: str = "workspaces/tmp") -> List[Path]:
    """Find all signal JSON files in the workspace."""
    base = Path(base_path)
    json_files = []
    
    if base.exists():
        # Look for signal files
        json_files.extend(base.glob("**/signals_*.json"))
        # Also look for classifier files if any
        json_files.extend(base.glob("**/classifiers_*.json"))
        # Look in subdirectories
        json_files.extend(base.glob("**/**/signals_*.json"))
        
    return sorted(json_files)


def extract_strategy_info(json_file: Path) -> Dict[str, Any]:
    """Extract strategy information from JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        strategies = metadata.get('strategies', {})
        
        # Get strategy info
        strategy_info = {}
        for strategy_key, info in strategies.items():
            # Extract strategy name and parameters
            parts = strategy_key.split('_')
            symbol = parts[0] if parts else 'UNKNOWN'
            strategy_name = '_'.join(parts[1:]) if len(parts) > 1 else strategy_key
            
            # Try to extract parameters
            params = {}
            if 'metadata' in info and 'params' in info['metadata']:
                params = info['metadata']['params']
            elif 'strategy_parameters' in metadata:
                params = metadata['strategy_parameters']
            
            strategy_info[strategy_key] = {
                'name': strategy_name,
                'symbol': symbol,
                'params': params,
                'total_bars': metadata.get('total_bars', 0),
                'changes': data.get('changes', [])
            }
        
        return strategy_info
        
    except Exception as e:
        logger.error(f"Failed to extract info from {json_file}: {e}")
        return {}


def migrate_file(json_file: Path, storage: HierarchicalParquetStorage) -> List[str]:
    """Migrate a single JSON file to hierarchical storage."""
    logger.info(f"Migrating {json_file}")
    
    saved_files = []
    
    try:
        # Extract strategy info
        strategy_info = extract_strategy_info(json_file)
        
        if not strategy_info:
            logger.warning(f"No strategy info found in {json_file}")
            return saved_files
        
        # Process each strategy
        for strategy_key, info in strategy_info.items():
            try:
                # Store in hierarchical structure
                storage_meta = storage.store_signal_data(
                    signal_changes=info['changes'],
                    strategy_name=info['name'],
                    parameters=info['params'],
                    metadata={
                        'total_bars': info['total_bars'],
                        'symbol': info['symbol'],
                        'timeframe': '1m',
                        'source_file': str(json_file)
                    }
                )
                
                saved_files.append(storage_meta.file_path)
                logger.info(f"  ✓ Migrated {info['name']} -> {storage_meta.file_path}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to migrate {strategy_key}: {e}")
    
    except Exception as e:
        logger.error(f"Failed to process {json_file}: {e}")
    
    return saved_files


def main():
    """Main migration function."""
    logger.info("Starting migration to hierarchical Parquet storage")
    
    # Initialize storage
    storage = HierarchicalParquetStorage(base_dir="./analytics_storage")
    
    # Find all JSON files
    json_files = find_signal_json_files()
    logger.info(f"Found {len(json_files)} JSON files to migrate")
    
    if not json_files:
        logger.info("No files to migrate")
        return
    
    # Migrate each file
    total_migrated = 0
    all_saved_files = []
    
    for json_file in json_files:
        saved_files = migrate_file(json_file, storage)
        all_saved_files.extend(saved_files)
        total_migrated += len(saved_files)
    
    # Summary
    logger.info("\nMigration Summary:")
    logger.info(f"  Total JSON files processed: {len(json_files)}")
    logger.info(f"  Total Parquet files created: {total_migrated}")
    
    # List created structure
    logger.info("\nCreated structure:")
    
    # Show signal structure
    signal_types = [d for d in storage.signals_dir.iterdir() if d.is_dir()]
    for type_dir in sorted(signal_types):
        parquet_files = list(type_dir.glob("*.parquet"))
        logger.info(f"  signals/{type_dir.name}/: {len(parquet_files)} files")
        
        # Show first few files
        for pf in sorted(parquet_files)[:3]:
            logger.info(f"    - {pf.name}")
        if len(parquet_files) > 3:
            logger.info(f"    ... and {len(parquet_files) - 3} more")
    
    # Show classifier structure
    classifier_types = [d for d in storage.classifiers_dir.iterdir() if d.is_dir()]
    for type_dir in sorted(classifier_types):
        parquet_files = list(type_dir.glob("*.parquet"))
        logger.info(f"  classifiers/{type_dir.name}/: {len(parquet_files)} files")
        
        # Show first few files
        for pf in sorted(parquet_files)[:3]:
            logger.info(f"    - {pf.name}")
        if len(parquet_files) > 3:
            logger.info(f"    ... and {len(parquet_files) - 3} more")
    
    logger.info("\nMigration complete!")
    
    # Optional: Show example of how to read the data
    if all_saved_files:
        logger.info("\nExample: Reading migrated data")
        example_file = all_saved_files[0]
        df, metadata = storage.load_signal_data(example_file)
        logger.info(f"  Loaded {example_file}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Metadata: {list(metadata.keys())}")


if __name__ == "__main__":
    main()