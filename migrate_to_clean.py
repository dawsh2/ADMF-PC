#!/usr/bin/env python3
"""
Migration script to transition from old architecture to clean BACKTEST.MD implementation.

This script:
1. Backs up existing files
2. Replaces old implementations with clean ones
3. Updates imports
4. Runs tests to verify
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import subprocess


def create_backup():
    """Create backup of current implementation."""
    backup_dir = Path(f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "src/core/coordinator/coordinator.py",
        "src/core/coordinator/managers.py",
        "src/core/coordinator/execution_modes.py",
        "src/core/coordinator/minimal_coordinator.py",
        "src/core/coordinator/simple_backtest_manager.py",
        "src/core/coordinator/backtest_manager.py",
        "src/core/containers/minimal_bootstrap.py",
        "src/execution/simple_backtest_engine.py",
        "main.py"
    ]
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            dest = backup_dir / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"Backed up: {file_path}")
    
    print(f"\nBackup created in: {backup_dir}")
    return backup_dir


def migrate_coordinator():
    """Replace old coordinator with clean implementation."""
    print("\nMigrating coordinator...")
    
    # Move clean implementations to proper locations
    if Path("src/core/coordinator/coordinator_clean.py").exists():
        shutil.move("src/core/coordinator/coordinator_clean.py",
                    "src/core/coordinator/coordinator.py")
        print("✓ Replaced coordinator.py with clean implementation")
    
    if Path("src/core/coordinator/managers_clean.py").exists():
        shutil.move("src/core/coordinator/managers_clean.py",
                    "src/core/coordinator/managers.py")
        print("✓ Replaced managers.py with clean implementation")


def cleanup_old_files():
    """Remove old duplicate files."""
    print("\nCleaning up old files...")
    
    files_to_remove = [
        # Coordinator duplicates
        "src/core/coordinator/minimal_coordinator.py",
        "src/core/coordinator/yaml_coordinator.py",
        "src/core/coordinator/simple_backtest_manager.py",
        "src/core/coordinator/backtest_manager.py",
        "src/core/coordinator/execution_modes.py",
        "src/core/coordinator/simple_types.py",
        "src/core/coordinator/types_no_pydantic.py",
        
        # Bootstrap duplicates
        "src/core/containers/minimal_bootstrap.py",
        
        # Execution duplicates
        "src/execution/simple_backtest_engine.py",
        "src/execution/backtest_broker_refactored.py",
        
        # Data duplicates
        "src/data/simple_loader.py",
        
        # Classifier duplicates
        "src/strategy/classifiers/enhanced_classifier_container.py",
        
        # Misplaced files
        "src/backtest/backtest_engine.py",
        "src/core/minimal_types.py"
    ]
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"✓ Removed: {file_path}")


def move_test_files():
    """Move test files to proper directories."""
    print("\nMoving test files...")
    
    # Create test directories if needed
    Path("tests/unit").mkdir(parents=True, exist_ok=True)
    Path("tests/integration").mkdir(parents=True, exist_ok=True)
    Path("examples").mkdir(parents=True, exist_ok=True)
    
    # Move test files from root
    test_files = list(Path(".").glob("test_*.py"))
    for test_file in test_files:
        if "integration" in test_file.name:
            dest = Path("tests/integration") / test_file.name
        else:
            dest = Path("tests/unit") / test_file.name
        
        shutil.move(str(test_file), str(dest))
        print(f"✓ Moved {test_file} to {dest}")
    
    # Move run files to examples
    run_files = list(Path(".").glob("run_*.py"))
    for run_file in run_files:
        dest = Path("examples") / run_file.name
        shutil.move(str(run_file), str(dest))
        print(f"✓ Moved {run_file} to {dest}")
    
    # Move example files
    example_files = list(Path(".").glob("example_*.py"))
    for example_file in example_files:
        dest = Path("examples") / example_file.name
        shutil.move(str(example_file), str(dest))
        print(f"✓ Moved {example_file} to {dest}")


def update_main_py():
    """Update main.py to use clean coordinator."""
    print("\nUpdating main.py...")
    
    main_content = '''#!/usr/bin/env python3
"""
ADMF-PC: Adaptive Dynamic Market Framework - Protocol Components

Main entry point - parses arguments and delegates to Coordinator.
"""

import asyncio
import argparse
import yaml
from typing import Dict, Any

from src.core.coordinator import Coordinator
from src.core.coordinator.types import WorkflowConfig, WorkflowType


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Dynamic Market Framework - Protocol Components'
    )
    
    # Core arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    # Execution mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'optimization', 'live'],
        default=None,
        help='Override execution mode from config'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'test', 'full'],
        default=None,
        help='Dataset to use (enables train/test splits)'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Limit data to first N bars'
    )
    
    # Logging arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def build_workflow_config(args: argparse.Namespace, base_config: Dict[str, Any]) -> WorkflowConfig:
    """Build workflow configuration from arguments and base config."""
    # Determine workflow type
    workflow_type = args.mode or base_config.get('workflow_type', 'backtest')
    
    # Create workflow config
    config = WorkflowConfig(
        workflow_type=WorkflowType(workflow_type),
        parameters=base_config.get('parameters', {}),
        data_config=base_config.get('data', {}),
        backtest_config=base_config.get('backtest', {}),
        optimization_config=base_config.get('optimization', {})
    )
    
    # Apply CLI overrides
    if args.dataset:
        config.data_config['dataset'] = args.dataset
    if args.bars:
        config.data_config['max_bars'] = args.bars
    
    return config


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Build workflow configuration
    workflow_config = build_workflow_config(args, base_config)
    
    # Execute workflow
    result = await coordinator.execute_workflow(workflow_config)
    
    # Shutdown
    await coordinator.shutdown()
    
    # Log result
    if result.success:
        logging.info("Workflow completed successfully")
        return 0
    else:
        logging.error("Workflow failed")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
'''
    
    with open("main.py", "w") as f:
        f.write(main_content)
    
    print("✓ Updated main.py")


def verify_imports():
    """Verify no circular imports remain."""
    print("\nVerifying imports...")
    
    # Check for circular imports
    result = subprocess.run(
        ["python3", "-c", "from src.core.coordinator import Coordinator; print('✓ Coordinator imports successfully')"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("✗ Import error:")
        print(result.stderr)
        return False
    
    return True


def main():
    """Run migration."""
    print("Starting migration to clean BACKTEST.MD architecture...")
    
    # Step 1: Create backup
    backup_dir = create_backup()
    
    try:
        # Step 2: Migrate coordinator
        migrate_coordinator()
        
        # Step 3: Cleanup old files
        cleanup_old_files()
        
        # Step 4: Move test files
        move_test_files()
        
        # Step 5: Update main.py
        update_main_py()
        
        # Step 6: Verify imports
        if verify_imports():
            print("\n✅ Migration completed successfully!")
            print(f"Backup available at: {backup_dir}")
        else:
            print("\n❌ Migration failed - check errors above")
            print(f"You can restore from: {backup_dir}")
            
    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        print(f"You can restore from: {backup_dir}")
        raise


if __name__ == "__main__":
    main()