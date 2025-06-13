# Workspace Migration Utilities
"""
Tools for migrating existing ADMF-PC workspaces to the new SQL format.
Handles conversion from UUID-based JSON workspaces to standardized SQL catalogs.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import hashlib
import re

from .workspace import AnalyticsWorkspace
from .exceptions import MigrationError, AnalyticsError


class WorkspaceMigrator:
    """Migrate existing workspaces to SQL format"""
    
    def __init__(self, source_path: Union[str, Path], destination_path: Union[str, Path]):
        """Initialize migrator
        
        Args:
            source_path: Path to existing workspace
            destination_path: Path for new SQL workspace
        """
        self.source_path = Path(source_path)
        self.destination_path = Path(destination_path)
        
        if not self.source_path.exists():
            raise MigrationError(f"Source workspace not found: {source_path}")
    
    def migrate(self, copy_files: bool = True, validate: bool = True) -> None:
        """Perform complete workspace migration
        
        Args:
            copy_files: Whether to copy signal/classifier files
            validate: Whether to validate migration success
        """
        try:
            print(f"Migrating workspace: {self.source_path} -> {self.destination_path}")
            
            # Create destination directory
            self.destination_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize new SQL workspace
            workspace = AnalyticsWorkspace(self.destination_path)
            
            # Detect source format and migrate
            if self._is_uuid_workspace():
                self._migrate_uuid_workspace(workspace, copy_files)
            elif self._is_legacy_json_workspace():
                self._migrate_legacy_json_workspace(workspace, copy_files)
            else:
                raise MigrationError(f"Unknown workspace format: {self.source_path}")
            
            # Validate if requested
            if validate:
                self._validate_migration(workspace)
            
            workspace.close()
            print("Migration completed successfully")
            
        except Exception as e:
            raise MigrationError(f"Migration failed: {e}")
    
    def _is_uuid_workspace(self) -> bool:
        """Check if source is UUID-based workspace"""
        # Look for UUID-style directory names and JSON files
        for item in self.source_path.iterdir():
            if item.is_dir() and re.match(r'^[0-9a-f-]{36}$', item.name):
                return True
        return False
    
    def _is_legacy_json_workspace(self) -> bool:
        """Check if source is legacy JSON workspace"""
        # Look for specific JSON files that indicate legacy format
        legacy_files = ['strategies.json', 'results.json', 'metadata.json']
        return any((self.source_path / f).exists() for f in legacy_files)
    
    def _migrate_uuid_workspace(self, workspace: AnalyticsWorkspace, copy_files: bool) -> None:
        """Migrate UUID-based workspace"""
        print("Detected UUID-based workspace format")
        
        # Find all strategy/result directories
        uuid_dirs = [d for d in self.source_path.iterdir() 
                    if d.is_dir() and re.match(r'^[0-9a-f-]{36}$', d.name)]
        
        if not uuid_dirs:
            raise MigrationError("No UUID directories found in workspace")
        
        # Create run record
        run_id = self._generate_run_id()
        run_info = self._extract_run_info_from_uuid_workspace(uuid_dirs)
        self._insert_run_record(workspace, run_id, run_info)
        
        # Process each UUID directory
        strategies_migrated = 0
        classifiers_migrated = 0
        
        for uuid_dir in uuid_dirs:
            try:
                result = self._migrate_uuid_directory(workspace, uuid_dir, run_id, copy_files)
                strategies_migrated += result.get('strategies', 0)
                classifiers_migrated += result.get('classifiers', 0)
            except Exception as e:
                print(f"Warning: Failed to migrate {uuid_dir.name}: {e}")
                continue
        
        print(f"Migrated {strategies_migrated} strategies and {classifiers_migrated} classifiers")
    
    def _migrate_legacy_json_workspace(self, workspace: AnalyticsWorkspace, copy_files: bool) -> None:
        """Migrate legacy JSON workspace"""
        print("Detected legacy JSON workspace format")
        
        # Load legacy JSON files
        strategies_file = self.source_path / 'strategies.json'
        if strategies_file.exists():
            with open(strategies_file, 'r') as f:
                strategies_data = json.load(f)
        else:
            strategies_data = {}
        
        # Create run record
        run_id = self._generate_run_id()
        run_info = self._extract_run_info_from_legacy(strategies_data)
        self._insert_run_record(workspace, run_id, run_info)
        
        # Migrate strategies
        self._migrate_legacy_strategies(workspace, strategies_data, run_id, copy_files)
    
    def _migrate_uuid_directory(self, workspace: AnalyticsWorkspace, uuid_dir: Path, 
                               run_id: str, copy_files: bool) -> Dict[str, int]:
        """Migrate single UUID directory"""
        migrated = {'strategies': 0, 'classifiers': 0}
        
        # Look for JSON files in the directory
        json_files = list(uuid_dir.glob('*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Determine file type and migrate appropriately
                if self._is_strategy_json(data):
                    self._migrate_strategy_from_uuid(workspace, data, uuid_dir, run_id, copy_files)
                    migrated['strategies'] += 1
                elif self._is_classifier_json(data):
                    self._migrate_classifier_from_uuid(workspace, data, uuid_dir, run_id, copy_files)
                    migrated['classifiers'] += 1
                
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue
        
        return migrated
    
    def _is_strategy_json(self, data: dict) -> bool:
        """Check if JSON represents strategy data"""
        strategy_indicators = ['strategy_type', 'parameters', 'sharpe_ratio', 'total_return']
        return any(key in data for key in strategy_indicators)
    
    def _is_classifier_json(self, data: dict) -> bool:
        """Check if JSON represents classifier data"""
        classifier_indicators = ['classifier_type', 'states', 'regime_counts', 'transitions']
        return any(key in data for key in classifier_indicators)
    
    def _migrate_strategy_from_uuid(self, workspace: AnalyticsWorkspace, data: dict, 
                                   uuid_dir: Path, run_id: str, copy_files: bool) -> None:
        """Migrate strategy from UUID directory"""
        
        # Generate strategy ID
        strategy_id = f"{data.get('strategy_type', 'unknown')}_{uuid_dir.name[:8]}"
        
        # Find signal file
        signal_files = list(uuid_dir.glob('*signals*.parquet')) + list(uuid_dir.glob('*signals*.json'))
        signal_file_path = None
        
        if signal_files and copy_files:
            source_signal_file = signal_files[0]
            # Create standardized path
            strategy_type = data.get('strategy_type', 'unknown')
            target_dir = self.destination_path / 'signals' / strategy_type
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_file = target_dir / f"{strategy_id}.parquet"
            
            # Convert JSON to Parquet if needed
            if source_signal_file.suffix == '.json':
                self._convert_json_signals_to_parquet(source_signal_file, target_file)
            else:
                shutil.copy2(source_signal_file, target_file)
            
            signal_file_path = f"signals/{strategy_type}/{strategy_id}.parquet"
        
        # Insert strategy record
        strategy_record = {
            'strategy_id': strategy_id,
            'run_id': run_id,
            'strategy_type': data.get('strategy_type', 'unknown'),
            'strategy_name': data.get('strategy_name', strategy_id),
            'parameters': json.dumps(data.get('parameters', {})),
            'signal_file_path': signal_file_path,
            'config_hash': self._hash_config(data.get('parameters', {})),
            
            # Performance metrics
            'total_return': data.get('total_return'),
            'annualized_return': data.get('annualized_return'),
            'volatility': data.get('volatility'),
            'sharpe_ratio': data.get('sharpe_ratio'),
            'max_drawdown': data.get('max_drawdown'),
            'total_trades': data.get('total_trades'),
            'win_rate': data.get('win_rate'),
            
            'created_at': datetime.now(),
            'processed_at': datetime.now()
        }
        
        # Insert into database
        self._insert_strategy_record(workspace, strategy_record)
    
    def _migrate_classifier_from_uuid(self, workspace: AnalyticsWorkspace, data: dict,
                                     uuid_dir: Path, run_id: str, copy_files: bool) -> None:
        """Migrate classifier from UUID directory"""
        
        # Generate classifier ID
        classifier_id = f"{data.get('classifier_type', 'unknown')}_{uuid_dir.name[:8]}"
        
        # Find states file
        states_files = list(uuid_dir.glob('*states*.parquet')) + list(uuid_dir.glob('*states*.json'))
        states_file_path = None
        
        if states_files and copy_files:
            source_states_file = states_files[0]
            classifier_type = data.get('classifier_type', 'unknown')
            target_dir = self.destination_path / 'classifiers' / classifier_type
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_file = target_dir / f"{classifier_id}.parquet"
            
            # Convert JSON to Parquet if needed
            if source_states_file.suffix == '.json':
                self._convert_json_states_to_parquet(source_states_file, target_file)
            else:
                shutil.copy2(source_states_file, target_file)
            
            states_file_path = f"classifiers/{classifier_type}/{classifier_id}.parquet"
        
        # Insert classifier record
        classifier_record = {
            'classifier_id': classifier_id,
            'run_id': run_id,
            'classifier_type': data.get('classifier_type', 'unknown'),
            'classifier_name': data.get('classifier_name', classifier_id),
            'parameters': json.dumps(data.get('parameters', {})),
            'states_file_path': states_file_path,
            'config_hash': self._hash_config(data.get('parameters', {})),
            
            # Classification characteristics
            'regime_counts': json.dumps(data.get('regime_counts', {})),
            'regime_durations': json.dumps(data.get('regime_durations', {})),
            'transition_matrix': json.dumps(data.get('transition_matrix', {})),
            
            'created_at': datetime.now(),
            'processed_at': datetime.now()
        }
        
        # Insert into database
        self._insert_classifier_record(workspace, classifier_record)
    
    def _convert_json_signals_to_parquet(self, source_file: Path, target_file: Path) -> None:
        """Convert JSON signals to Parquet format"""
        try:
            with open(source_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame with standardized schema
            if isinstance(data, dict) and 'signals' in data:
                signals_data = data['signals']
            else:
                signals_data = data
            
            # Create DataFrame with standard columns
            df_data = []
            for idx, signal in signals_data.items() if isinstance(signals_data, dict) else enumerate(signals_data):
                df_data.append({
                    'bar_idx': int(idx),
                    'signal': signal,
                    'timestamp': None  # Will be filled later if available
                })
            
            df = pd.DataFrame(df_data)
            df.to_parquet(target_file, index=False)
            
        except Exception as e:
            raise MigrationError(f"Failed to convert signals JSON to Parquet: {e}")
    
    def _convert_json_states_to_parquet(self, source_file: Path, target_file: Path) -> None:
        """Convert JSON states to Parquet format"""
        try:
            with open(source_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame with standardized schema
            df_data = []
            for idx, state in data.items() if isinstance(data, dict) else enumerate(data):
                df_data.append({
                    'bar_idx': int(idx),
                    'regime': state,
                    'timestamp': None
                })
            
            df = pd.DataFrame(df_data)
            df.to_parquet(target_file, index=False)
            
        except Exception as e:
            raise MigrationError(f"Failed to convert states JSON to Parquet: {e}")
    
    def _generate_run_id(self) -> str:
        """Generate run ID from workspace name and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_name = self.source_path.name
        return f"{timestamp}_{workspace_name}"
    
    def _extract_run_info_from_uuid_workspace(self, uuid_dirs: List[Path]) -> dict:
        """Extract run information from UUID workspace"""
        return {
            'workflow_type': 'legacy_migration',
            'total_strategies': len(uuid_dirs),
            'symbols': ['SPY'],  # Default - update if metadata available
            'timeframes': ['1m'],  # Default
            'status': 'completed'
        }
    
    def _extract_run_info_from_legacy(self, strategies_data: dict) -> dict:
        """Extract run information from legacy JSON"""
        return {
            'workflow_type': 'legacy_migration',
            'total_strategies': len(strategies_data),
            'symbols': ['SPY'],  # Default
            'timeframes': ['1m'],  # Default
            'status': 'completed'
        }
    
    def _hash_config(self, config: dict) -> str:
        """Generate hash for configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _insert_run_record(self, workspace: AnalyticsWorkspace, run_id: str, run_info: dict) -> None:
        """Insert run record into database"""
        workspace.conn.execute("""
            INSERT OR REPLACE INTO runs (
                run_id, created_at, workflow_type, symbols, timeframes,
                total_strategies, status, workspace_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id,
            datetime.now(),
            run_info.get('workflow_type', 'unknown'),
            run_info.get('symbols', []),
            run_info.get('timeframes', []),
            run_info.get('total_strategies', 0),
            run_info.get('status', 'completed'),
            str(self.destination_path)
        ])
    
    def _insert_strategy_record(self, workspace: AnalyticsWorkspace, record: dict) -> None:
        """Insert strategy record into database"""
        columns = ', '.join(record.keys())
        placeholders = ', '.join(['?' for _ in record])
        
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO strategies ({columns})
            VALUES ({placeholders})
        """, list(record.values()))
    
    def _insert_classifier_record(self, workspace: AnalyticsWorkspace, record: dict) -> None:
        """Insert classifier record into database"""
        columns = ', '.join(record.keys())
        placeholders = ', '.join(['?' for _ in record])
        
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO classifiers ({columns})
            VALUES ({placeholders})
        """, list(record.values()))
    
    def _migrate_legacy_strategies(self, workspace: AnalyticsWorkspace, strategies_data: dict,
                                  run_id: str, copy_files: bool) -> None:
        """Migrate strategies from legacy JSON format"""
        for strategy_id, data in strategies_data.items():
            try:
                self._migrate_strategy_from_uuid(workspace, data, self.source_path, run_id, copy_files)
            except Exception as e:
                print(f"Warning: Failed to migrate strategy {strategy_id}: {e}")
    
    def _validate_migration(self, workspace: AnalyticsWorkspace) -> None:
        """Validate migration was successful"""
        try:
            # Check that tables have data
            runs = workspace.sql("SELECT COUNT(*) as count FROM runs")
            strategies = workspace.sql("SELECT COUNT(*) as count FROM strategies")
            
            if runs.iloc[0]['count'] == 0:
                raise MigrationError("No runs found after migration")
            
            if strategies.iloc[0]['count'] == 0:
                raise MigrationError("No strategies found after migration")
            
            print(f"Validation successful: {runs.iloc[0]['count']} runs, {strategies.iloc[0]['count']} strategies")
            
        except Exception as e:
            raise MigrationError(f"Migration validation failed: {e}")


def migrate_workspace(source_path: Union[str, Path], destination_path: Union[str, Path],
                     copy_files: bool = True, validate: bool = True) -> None:
    """Convenience function to migrate a workspace
    
    Args:
        source_path: Path to existing workspace
        destination_path: Path for new SQL workspace
        copy_files: Whether to copy signal/classifier files
        validate: Whether to validate migration success
    """
    migrator = WorkspaceMigrator(source_path, destination_path)
    migrator.migrate(copy_files=copy_files, validate=validate)


def setup_workspace(workspace_path: Union[str, Path]) -> AnalyticsWorkspace:
    """Setup new empty SQL workspace
    
    Args:
        workspace_path: Path for new workspace
        
    Returns:
        Initialized AnalyticsWorkspace
    """
    workspace_path = Path(workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    return AnalyticsWorkspace(workspace_path)


def find_workspaces(search_path: Union[str, Path]) -> List[Path]:
    """Find all workspaces in a directory
    
    Args:
        search_path: Directory to search
        
    Returns:
        List of workspace paths
    """
    search_path = Path(search_path)
    workspaces = []
    
    for item in search_path.iterdir():
        if item.is_dir():
            # Check for SQL workspace
            if (item / 'analytics.duckdb').exists():
                workspaces.append(item)
            # Check for legacy workspace indicators
            elif any((item / f).exists() for f in ['strategies.json', 'results.json']):
                workspaces.append(item)
            # Check for UUID directories
            elif any(d.is_dir() and re.match(r'^[0-9a-f-]{36}$', d.name) 
                    for d in item.iterdir() if d.is_dir()):
                workspaces.append(item)
    
    return sorted(workspaces)