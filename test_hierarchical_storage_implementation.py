#!/usr/bin/env python3
"""
Test script for hierarchical Parquet storage implementation.

This script:
1. Runs a simple signal generation workflow
2. Verifies that signals are stored in the hierarchical structure
3. Checks that Parquet files are created correctly
4. Lists the created files and their structure
"""

import os
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator import Coordinator
from src.analytics.storage.hierarchical_parquet_storage import HierarchicalParquetStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test_workflow():
    """Run the test workflow with hierarchical storage."""
    logger.info("=" * 80)
    logger.info("Testing Hierarchical Parquet Storage")
    logger.info("=" * 80)
    
    # Load config
    config_path = "config/test_hierarchical_storage.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config: {config_path}")
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Run the workflow
    logger.info("\nRunning signal generation workflow...")
    try:
        result = coordinator.run_topology(config)
        logger.info("Workflow completed successfully")
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return False
    
    return True


def verify_hierarchical_storage():
    """Verify that the hierarchical storage was created correctly."""
    logger.info("\n" + "=" * 80)
    logger.info("Verifying Hierarchical Storage Structure")
    logger.info("=" * 80)
    
    storage_dir = Path("./analytics_storage")
    
    if not storage_dir.exists():
        logger.error(f"Storage directory does not exist: {storage_dir}")
        return False
    
    # Check for signals directory
    signals_dir = storage_dir / "signals"
    if not signals_dir.exists():
        logger.error(f"Signals directory does not exist: {signals_dir}")
        return False
    
    # Check for classifiers directory
    classifiers_dir = storage_dir / "classifiers"
    if not classifiers_dir.exists():
        logger.warning(f"Classifiers directory does not exist: {classifiers_dir} (may be expected if no classifiers)")
    
    # List signal types
    logger.info("\nSignal Storage Structure:")
    signal_types = [d for d in signals_dir.iterdir() if d.is_dir()]
    
    total_files = 0
    for type_dir in sorted(signal_types):
        logger.info(f"\n  {type_dir.name}/")
        
        # List Parquet files
        parquet_files = list(type_dir.glob("*.parquet"))
        total_files += len(parquet_files)
        
        for pf in sorted(parquet_files):
            logger.info(f"    - {pf.name}")
            
            # Try to read the file
            try:
                storage = HierarchicalParquetStorage(base_dir="./analytics_storage")
                df, metadata = storage.load_signal_data(str(pf))
                logger.info(f"      Shape: {df.shape}, Changes: {len(df)}")
                logger.info(f"      Metadata keys: {list(metadata.keys())}")
                
                # Show first few changes
                if len(df) > 0:
                    logger.info(f"      First change: bar {df.iloc[0]['bar_idx']}, signal {df.iloc[0]['signal']}")
                    if len(df) > 1:
                        logger.info(f"      Last change: bar {df.iloc[-1]['bar_idx']}, signal {df.iloc[-1]['signal']}")
                        
            except Exception as e:
                logger.error(f"      Failed to read: {e}")
        
        # Check for index file
        index_file = type_dir / "index.json"
        if index_file.exists():
            logger.info(f"    - index.json ✓")
        else:
            logger.warning(f"    - index.json ✗ (missing)")
    
    # List classifier types if any
    if classifiers_dir.exists():
        logger.info("\nClassifier Storage Structure:")
        classifier_types = [d for d in classifiers_dir.iterdir() if d.is_dir()]
        
        for type_dir in sorted(classifier_types):
            logger.info(f"\n  {type_dir.name}/")
            
            # List Parquet files
            parquet_files = list(type_dir.glob("*.parquet"))
            total_files += len(parquet_files)
            
            for pf in sorted(parquet_files):
                logger.info(f"    - {pf.name}")
    
    logger.info(f"\nTotal Parquet files created: {total_files}")
    
    return total_files > 0


def compare_with_old_storage():
    """Compare with old JSON storage if it exists."""
    logger.info("\n" + "=" * 80)
    logger.info("Comparison with Old Storage")
    logger.info("=" * 80)
    
    # Check if old storage exists
    old_storage = Path("./workspaces/tmp")
    if old_storage.exists():
        json_files = list(old_storage.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} JSON files in old storage")
        
        # Show file sizes
        total_json_size = sum(f.stat().st_size for f in json_files)
        logger.info(f"Total JSON size: {total_json_size / 1024:.1f} KB")
    else:
        logger.info("No old JSON storage found")
    
    # Check new storage
    new_storage = Path("./analytics_storage")
    if new_storage.exists():
        parquet_files = list(new_storage.glob("**/*.parquet"))
        logger.info(f"Found {len(parquet_files)} Parquet files in new storage")
        
        # Show file sizes
        total_parquet_size = sum(f.stat().st_size for f in parquet_files)
        logger.info(f"Total Parquet size: {total_parquet_size / 1024:.1f} KB")
        
        if old_storage.exists() and json_files:
            compression = (1 - total_parquet_size / total_json_size) * 100
            logger.info(f"Storage compression: {compression:.1f}%")


def main():
    """Main test function."""
    logger.info("Starting hierarchical storage test")
    
    # Run the workflow
    if not run_test_workflow():
        logger.error("Workflow failed, skipping verification")
        return 1
    
    # Verify storage
    if not verify_hierarchical_storage():
        logger.error("Storage verification failed")
        return 1
    
    # Compare with old storage
    compare_with_old_storage()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Hierarchical storage test completed successfully!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())