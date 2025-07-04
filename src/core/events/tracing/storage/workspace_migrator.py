# Workspace Migration Tool
"""
Migrates workspaces from old UUID-based format to new standardized format
with support for grid search results and sparse storage.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import shutil
from collections import defaultdict
import hashlib
import re

from .sparse_storage import SparseSignalStorage, SparseClassifierStorage


class StrategyGrouper:
    """Groups strategies by type and generates consistent IDs"""
    
    @staticmethod
    def extract_strategy_type(strategy_id: str) -> str:
        """Extract strategy type from ID"""
        # Common patterns
        patterns = [
            (r'ma_crossover', 'ma_crossover'),
            (r'momentum', 'momentum'),
            (r'mean_reversion', 'mean_reversion'),
            (r'rsi', 'rsi_strategy'),
            (r'macd', 'macd_strategy'),
            (r'breakout', 'breakout_strategy'),
            (r'bollinger', 'mean_reversion'),
            (r'sma', 'ma_crossover')
        ]
        
        strategy_lower = strategy_id.lower()
        for pattern, strategy_type in patterns:
            if pattern in strategy_lower:
                return strategy_type
        
        return 'unknown'
    
    @staticmethod
    def extract_parameters(strategy_id: str, strategy_type: str) -> Dict[str, Any]:
        """Extract parameters from strategy ID"""
        # Try to parse numbers from the ID
        numbers = re.findall(r'\d+\.?\d*', strategy_id)
        
        # Map to standard parameters based on strategy type
        params = {}
        if strategy_type == 'ma_crossover' and len(numbers) >= 2:
            params['fast_period'] = int(numbers[0])
            params['slow_period'] = int(numbers[1])
            if len(numbers) > 2:
                params['stop_loss_pct'] = float(numbers[2])
        elif strategy_type == 'momentum' and numbers:
            if len(numbers) >= 1:
                params['sma_period'] = int(numbers[0])
            if len(numbers) >= 3:
                params['rsi_threshold_long'] = int(numbers[1])
                params['rsi_threshold_short'] = int(numbers[2])
        elif strategy_type == 'rsi_strategy' and numbers:
            if len(numbers) >= 1:
                params['period'] = int(numbers[0])
            if len(numbers) >= 3:
                params['oversold'] = int(numbers[1])
                params['overbought'] = int(numbers[2])
        
        return params
    
    @staticmethod
    def generate_strategy_filename(strategy_type: str, params: Dict[str, Any]) -> str:
        """Generate consistent filename for strategy"""
        # Create short parameter string
        if strategy_type == 'ma_crossover':
            param_str = f"ma_{params.get('fast_period', 0)}_{params.get('slow_period', 0)}"
            if 'stop_loss_pct' in params:
                param_str += f"_sl_{params['stop_loss_pct']}"
        elif strategy_type == 'momentum':
            param_str = f"mom_{params.get('sma_period', 0)}"
            if 'rsi_threshold_long' in params:
                param_str += f"_{params['rsi_threshold_long']}_{params['rsi_threshold_short']}"
        else:
            # Generic parameter encoding
            param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        
        # Add hash for uniqueness
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        
        return f"{param_str}_{param_hash}"


class WorkspaceMigrator:
    """Migrate from old format to new grid-search optimized format"""
    
    def __init__(self, old_root: Path, new_root: Path):
        self.old_root = Path(old_root)
        self.new_root = Path(new_root)
        self.new_root.mkdir(parents=True, exist_ok=True)
        self.strategy_grouper = StrategyGrouper()
        
    def migrate_all(self, dry_run: bool = False) -> List[Tuple[str, str]]:
        """Migrate all workspaces
        
        Returns:
            List of (old_path, new_path) tuples
        """
        migrations = []
        
        for old_dir in self.old_root.iterdir():
            if not old_dir.is_dir():
                continue
                
            # Skip if already in new format
            if re.match(r'^\d{8}_\d{6}_', old_dir.name):
                continue
            
            try:
                if dry_run:
                    new_path = self._plan_migration(old_dir)
                    migrations.append((str(old_dir), str(new_path)))
                else:
                    new_path = self.migrate_workspace(old_dir.name)
                    migrations.append((str(old_dir), str(new_path)))
            except Exception as e:
                print(f"Error migrating {old_dir}: {e}")
        
        return migrations
    
    def _plan_migration(self, old_path: Path) -> Path:
        """Plan migration without executing"""
        workspace_info = self._analyze_workspace(old_path)
        new_name = self._generate_directory_name(workspace_info)
        return self.new_root / new_name
    
    def migrate_workspace(self, workspace_id: str) -> Path:
        """Migrate a single workspace"""
        old_path = self.old_root / workspace_id
        
        if not old_path.exists():
            raise ValueError(f"Workspace {workspace_id} not found")
        
        # Analyze workspace
        workspace_info = self._analyze_workspace(old_path)
        
        # Generate new directory name
        new_name = self._generate_directory_name(workspace_info)
        new_path = self.new_root / new_name
        
        if new_path.exists():
            print(f"Workspace {new_name} already exists, skipping")
            return new_path
        
        new_path.mkdir(parents=True, exist_ok=True)
        
        # Migrate components
        self._migrate_signals(old_path, new_path, workspace_info)
        self._migrate_classifiers(old_path, new_path, workspace_info)
        self._consolidate_events(old_path, new_path)
        self._create_manifest(workspace_info, new_path)
        self._create_indices(new_path, workspace_info)
        
        print(f"Migrated {workspace_id} -> {new_name}")
        return new_path
    
    def _analyze_workspace(self, old_path: Path) -> Dict[str, Any]:
        """Analyze old workspace structure"""
        info = {
            'workspace_id': old_path.name,
            'created_at': datetime.fromtimestamp(old_path.stat().st_mtime),
            'type': 'unknown',
            'symbols': set(),
            'strategies': {},
            'classifiers': {},
            'total_bars': 0
        }
        
        # Load metadata if exists
        metadata_path = old_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                info['created_at'] = datetime.fromisoformat(metadata.get('created_at', info['created_at'].isoformat()))
        
        # Check for signal files in tmp directory
        tmp_pattern = old_path.parent / 'tmp' / '*' / f'*{old_path.name}*.json'
        signal_files = list(old_path.parent.glob(str(tmp_pattern)))
        
        # Analyze strategies
        for signal_file in signal_files:
            if 'signals_' in signal_file.name:
                self._analyze_signal_file(signal_file, info)
        
        # Also check for strategy containers
        for container_dir in old_path.glob('strategy_*'):
            info['strategies'][container_dir.name] = {
                'type': 'unknown',
                'container_id': container_dir.name
            }
        
        # Check for classifiers
        for container_dir in old_path.glob('classifier_*'):
            info['classifiers'][container_dir.name] = {
                'type': 'unknown',
                'container_id': container_dir.name
            }
        
        # Determine workspace type
        if len(info['strategies']) > 5:
            info['type'] = 'grid_search'
        elif info['strategies']:
            info['type'] = 'optimization'
        else:
            info['type'] = 'backtest'
        
        return info
    
    def _analyze_signal_file(self, signal_file: Path, info: Dict[str, Any]):
        """Analyze a signal JSON file"""
        try:
            with open(signal_file) as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            info['total_bars'] = max(info['total_bars'], metadata.get('total_bars', 0))
            
            # Extract symbols
            for change in data.get('changes', []):
                if 'sym' in change:
                    info['symbols'].add(change['sym'])
            
            # Extract strategies
            for strategy_id, strategy_data in metadata.get('strategies', {}).items():
                strategy_type = self.strategy_grouper.extract_strategy_type(strategy_id)
                params = metadata.get('strategy_parameters', {})
                
                info['strategies'][strategy_id] = {
                    'type': strategy_type,
                    'data': strategy_data,
                    'params': params,
                    'signal_file': str(signal_file)
                }
                
        except Exception as e:
            print(f"Error analyzing {signal_file}: {e}")
    
    def _generate_directory_name(self, workspace_info: Dict[str, Any]) -> str:
        """Generate new directory name"""
        timestamp = workspace_info['created_at']
        workflow_type = workspace_info['type']
        symbols = '_'.join(sorted(workspace_info['symbols'])) or 'UNKNOWN'
        
        # Create identifier
        if workflow_type == 'grid_search':
            num_strategies = len(workspace_info['strategies'])
            identifier = f"{num_strategies}strategies"
        else:
            identifier = workspace_info['workspace_id'][:8]
        
        # Format: YYYYMMDD_HHMMSS_type_symbols_identifier
        name = f"{timestamp:%Y%m%d_%H%M%S}_{workflow_type}_{symbols}_{identifier}"
        
        # Sanitize name
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        return name
    
    def _migrate_signals(self, old_path: Path, new_path: Path, workspace_info: Dict[str, Any]):
        """Migrate signal files to new format"""
        signals_dir = new_path / 'signals'
        signals_dir.mkdir(exist_ok=True)
        
        # Group strategies by type
        strategies_by_type = defaultdict(list)
        
        for strategy_id, strategy_info in workspace_info['strategies'].items():
            strategy_type = strategy_info['type']
            strategies_by_type[strategy_type].append((strategy_id, strategy_info))
        
        # Process each strategy
        for strategy_type, strategies in strategies_by_type.items():
            type_dir = signals_dir / strategy_type
            type_dir.mkdir(exist_ok=True)
            
            for strategy_id, strategy_info in strategies:
                # Load signal data
                if 'signal_file' in strategy_info and Path(strategy_info['signal_file']).exists():
                    with open(strategy_info['signal_file']) as f:
                        signal_data = json.load(f)
                    
                    # Convert to sparse DataFrame
                    changes = signal_data.get('changes', [])
                    total_bars = signal_data.get('metadata', {}).get('total_bars', workspace_info['total_bars'])
                    
                    df = SparseSignalStorage.from_json_changes(
                        changes, 
                        total_bars,
                        strategy_info
                    )
                    
                    # Generate filename
                    params = strategy_info.get('params', {})
                    filename = self.strategy_grouper.generate_strategy_filename(strategy_type, params)
                    
                    # Save as Parquet
                    parquet_path = type_dir / f"{filename}.parquet"
                    SparseSignalStorage.to_parquet(df, parquet_path)
                    
                    # Update strategy info
                    strategy_info['parquet_file'] = str(parquet_path.relative_to(new_path))
                    strategy_info['signal_changes'] = len(changes)
                    strategy_info['compression_ratio'] = df.attrs.get('compression_ratio', 0)
    
    def _migrate_classifiers(self, old_path: Path, new_path: Path, workspace_info: Dict[str, Any]):
        """Migrate classifier files to new format"""
        # Placeholder - implement classifier migration
        classifiers_dir = new_path / 'classifiers'
        classifiers_dir.mkdir(exist_ok=True)
    
    def _consolidate_events(self, old_path: Path, new_path: Path):
        """Consolidate event files"""
        events_dir = new_path / 'events'
        events_dir.mkdir(exist_ok=True)
        
        # Copy event files
        for event_file in old_path.rglob('events.jsonl'):
            # Could convert to Parquet for efficiency
            target = events_dir / f"{event_file.parent.name}_events.jsonl"
            shutil.copy2(event_file, target)
    
    def _create_manifest(self, workspace_info: Dict[str, Any], new_path: Path):
        """Create new manifest file"""
        manifest = {
            'run_id': workspace_info['workspace_id'],
            'created_at': workspace_info['created_at'].isoformat(),
            'migrated_at': datetime.now().isoformat(),
            'workflow': {
                'type': workspace_info['type'],
                'format_version': '2.0'
            },
            'data': {
                'symbols': list(workspace_info['symbols']),
                'total_bars': workspace_info['total_bars']
            },
            'summary': {
                'total_strategies': len(workspace_info['strategies']),
                'total_classifiers': len(workspace_info['classifiers'])
            },
            'storage': {
                'format': 'sparse_parquet',
                'compression': 'snappy'
            }
        }
        
        with open(new_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _create_indices(self, new_path: Path, workspace_info: Dict[str, Any]):
        """Create index files for fast scanning"""
        # Create signals index
        signals_index = {
            'strategies': defaultdict(lambda: {'total': 0, 'files': {}})
        }
        
        signals_dir = new_path / 'signals'
        if signals_dir.exists():
            for strategy_type_dir in signals_dir.iterdir():
                if strategy_type_dir.is_dir():
                    strategy_type = strategy_type_dir.name
                    
                    for parquet_file in strategy_type_dir.glob('*.parquet'):
                        # Extract info from filename
                        file_id = parquet_file.stem
                        
                        signals_index['strategies'][strategy_type]['files'][file_id] = {
                            'file': f"{strategy_type}/{parquet_file.name}",
                            'params': {}  # Could extract from metadata
                        }
                    
                    signals_index['strategies'][strategy_type]['total'] = len(
                        signals_index['strategies'][strategy_type]['files']
                    )
        
        # Convert defaultdict to regular dict for JSON serialization
        signals_index['strategies'] = dict(signals_index['strategies'])
        
        with open(new_path / 'signals' / 'index.json', 'w') as f:
            json.dump(signals_index, f, indent=2)