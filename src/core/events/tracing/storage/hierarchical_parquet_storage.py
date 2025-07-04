"""
Hierarchical Parquet Storage Manager

Implements clean hierarchical storage structure for signals and classifiers:
- signals/
  - momentum/
  - ma_crossover/
  - mean_reversion/
- classifiers/
  - regime/
  - volatility/

Each file is named with parameter hash for uniqueness.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, asdict

from .sparse_storage import SparseSignalStorage, SparseClassifierStorage

logger = logging.getLogger(__name__)


@dataclass
class SignalStorageMetadata:
    """Metadata for a signal storage file"""
    strategy_type: str
    strategy_name: str
    parameters: Dict[str, Any]
    symbol: str
    timeframe: str
    total_bars: int
    signal_changes: int
    compression_ratio: float
    created_at: str
    file_hash: str
    file_path: str


@dataclass
class ClassifierStorageMetadata:
    """Metadata for a classifier storage file"""
    classifier_type: str
    classifier_name: str
    parameters: Dict[str, Any]
    total_bars: int
    regime_changes: int
    compression_ratio: float
    avg_regime_duration: float
    created_at: str
    file_hash: str
    file_path: str


class HierarchicalParquetStorage:
    """
    Manages hierarchical Parquet storage for signals and classifiers.
    
    Creates clean directory structure with hash-based filenames and index files.
    """
    
    def __init__(self, base_dir: str = "./analytics_storage"):
        """Initialize storage manager with base directory."""
        self.base_dir = Path(base_dir)
        self.signals_dir = self.base_dir / "signals"
        self.classifiers_dir = self.base_dir / "classifiers"
        
        # Create base directories
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.classifiers_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage helpers
        self.signal_storage = SparseSignalStorage()
        self.classifier_storage = SparseClassifierStorage()
        
        logger.info(f"Initialized hierarchical storage at {self.base_dir}")
    
    def _generate_param_hash(self, params: Dict[str, Any]) -> str:
        """Generate a hash from parameters for unique filename."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()[:8]
    
    def _get_strategy_type(self, strategy_name: str, strategy_type: Optional[str] = None) -> str:
        """Extract strategy type from name or type field."""
        if strategy_type:
            return strategy_type
            
        # Common patterns in strategy names
        if 'momentum' in strategy_name.lower():
            return 'momentum'
        elif 'ma' in strategy_name.lower() or 'crossover' in strategy_name.lower():
            return 'ma_crossover'
        elif 'mean_reversion' in strategy_name.lower() or 'bollinger' in strategy_name.lower():
            return 'mean_reversion'
        elif 'rsi' in strategy_name.lower():
            return 'rsi'
        elif 'macd' in strategy_name.lower():
            return 'macd'
        elif 'breakout' in strategy_name.lower():
            return 'breakout'
        else:
            return 'other'
    
    def _get_classifier_type(self, classifier_name: str, classifier_type: Optional[str] = None) -> str:
        """Extract classifier type from name or type field."""
        if classifier_type:
            return classifier_type
            
        # Common patterns in classifier names
        if 'regime' in classifier_name.lower() or 'hmm' in classifier_name.lower():
            return 'regime'
        elif 'volatility' in classifier_name.lower() or 'vol' in classifier_name.lower():
            return 'volatility'
        elif 'trend' in classifier_name.lower():
            return 'trend'
        elif 'market_state' in classifier_name.lower():
            return 'market_state'
        else:
            return 'other'
    
    def _create_signal_filename(self, strategy_name: str, params: Dict[str, Any]) -> str:
        """Create filename for signal storage."""
        # Extract key parameters for readable filename
        param_parts = []
        
        # Common strategy parameters
        if 'fast_period' in params and 'slow_period' in params:
            param_parts.extend([str(params['fast_period']), str(params['slow_period'])])
        elif 'period' in params:
            param_parts.append(str(params['period']))
        elif 'rsi_period' in params:
            param_parts.append(str(params['rsi_period']))
        
        # Add other significant parameters
        if 'threshold' in params:
            param_parts.append(f"th_{params['threshold']}")
        if 'stop_loss' in params:
            param_parts.append(f"sl_{params['stop_loss']}")
            
        # Generate hash for uniqueness
        param_hash = self._generate_param_hash(params)
        
        # Build filename
        if param_parts:
            filename = f"{strategy_name}_{'_'.join(param_parts)}_{param_hash}.parquet"
        else:
            filename = f"{strategy_name}_{param_hash}.parquet"
            
        return filename
    
    def _create_classifier_filename(self, classifier_name: str, params: Dict[str, Any]) -> str:
        """Create filename for classifier storage."""
        # Extract key parameters
        param_parts = []
        
        if 'n_states' in params:
            param_parts.append(f"{params['n_states']}state")
        if 'threshold' in params:
            param_parts.append(f"th_{params['threshold']}")
        if 'window' in params:
            param_parts.append(f"w{params['window']}")
            
        # Generate hash
        param_hash = self._generate_param_hash(params)
        
        # Build filename
        if param_parts:
            filename = f"{classifier_name}_{'_'.join(param_parts)}_{param_hash}.parquet"
        else:
            filename = f"{classifier_name}_{param_hash}.parquet"
            
        return filename
    
    def store_signal_data(
        self,
        signal_changes: List[Dict[str, Any]],
        strategy_name: str,
        strategy_type: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> SignalStorageMetadata:
        """
        Store signal data in hierarchical structure.
        
        Args:
            signal_changes: List of signal change events
            strategy_name: Name of the strategy
            strategy_type: Optional type override
            parameters: Strategy parameters
            metadata: Additional metadata
            
        Returns:
            Storage metadata with file location
        """
        if not parameters:
            parameters = {}
            
        # Determine strategy type and create directory
        strat_type = self._get_strategy_type(strategy_name, strategy_type)
        type_dir = self.signals_dir / strat_type
        type_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = self._create_signal_filename(strategy_name, parameters)
        file_path = type_dir / filename
        
        # Extract metadata
        total_bars = metadata.get('total_bars', 0)
        symbol = metadata.get('symbol', 'UNKNOWN')
        timeframe = metadata.get('timeframe', '1m')
        
        # Convert to DataFrame
        df = self.signal_storage.from_json_changes(
            signal_changes,
            total_bars,
            {
                'strategy_name': strategy_name,
                'parameters': parameters
            }
        )
        
        # Calculate compression
        compression_ratio = len(signal_changes) / total_bars if total_bars > 0 else 0
        
        # Save to Parquet
        storage_metadata = {
            'strategy_name': strategy_name,
            'strategy_type': strat_type,
            'parameters': parameters,
            'symbol': symbol,
            'timeframe': timeframe,
            'created_at': datetime.now().isoformat()
        }
        
        self.signal_storage.to_parquet(df, file_path, metadata=storage_metadata)
        
        # Create metadata object
        storage_meta = SignalStorageMetadata(
            strategy_type=strat_type,
            strategy_name=strategy_name,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total_bars,
            signal_changes=len(signal_changes),
            compression_ratio=compression_ratio,
            created_at=storage_metadata['created_at'],
            file_hash=self._generate_param_hash(parameters),
            file_path=str(file_path)
        )
        
        # Update index
        self._update_signal_index(strat_type, storage_meta)
        
        logger.info(f"Stored {len(signal_changes)} signal changes to {file_path}")
        logger.info(f"Compression: {compression_ratio:.2%} ({len(signal_changes)}/{total_bars} bars)")
        
        return storage_meta
    
    def store_classifier_data(
        self,
        regime_changes: List[Dict[str, Any]],
        classifier_name: str,
        classifier_type: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> ClassifierStorageMetadata:
        """
        Store classifier data in hierarchical structure.
        
        Args:
            regime_changes: List of regime change events
            classifier_name: Name of the classifier
            classifier_type: Optional type override
            parameters: Classifier parameters
            metadata: Additional metadata
            
        Returns:
            Storage metadata with file location
        """
        if not parameters:
            parameters = {}
            
        # Determine classifier type and create directory
        class_type = self._get_classifier_type(classifier_name, classifier_type)
        type_dir = self.classifiers_dir / class_type
        type_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = self._create_classifier_filename(classifier_name, parameters)
        file_path = type_dir / filename
        
        # Extract metadata
        total_bars = metadata.get('total_bars', 0)
        
        # Convert to DataFrame
        df = self.classifier_storage.from_regime_changes(
            regime_changes,
            total_bars,
            {
                'classifier_name': classifier_name,
                'parameters': parameters
            }
        )
        
        # Calculate statistics
        compression_ratio = len(regime_changes) / total_bars if total_bars > 0 else 0
        avg_duration = df.attrs.get('avg_regime_duration', 0)
        
        # Save to Parquet
        storage_metadata = {
            'classifier_name': classifier_name,
            'classifier_type': class_type,
            'parameters': parameters,
            'created_at': datetime.now().isoformat()
        }
        
        self.classifier_storage.to_parquet(df, file_path, metadata=storage_metadata)
        
        # Create metadata object
        storage_meta = ClassifierStorageMetadata(
            classifier_type=class_type,
            classifier_name=classifier_name,
            parameters=parameters,
            total_bars=total_bars,
            regime_changes=len(regime_changes),
            compression_ratio=compression_ratio,
            avg_regime_duration=avg_duration,
            created_at=storage_metadata['created_at'],
            file_hash=self._generate_param_hash(parameters),
            file_path=str(file_path)
        )
        
        # Update index
        self._update_classifier_index(class_type, storage_meta)
        
        logger.info(f"Stored {len(regime_changes)} regime changes to {file_path}")
        logger.info(f"Compression: {compression_ratio:.2%}, Avg duration: {avg_duration:.1f} bars")
        
        return storage_meta
    
    def _update_signal_index(self, strategy_type: str, metadata: SignalStorageMetadata) -> None:
        """Update index.json for a strategy type."""
        index_path = self.signals_dir / strategy_type / "index.json"
        
        # Load existing index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {
                'strategy_type': strategy_type,
                'files': [],
                'updated_at': datetime.now().isoformat()
            }
        
        # Add new entry
        index['files'].append(asdict(metadata))
        index['updated_at'] = datetime.now().isoformat()
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _update_classifier_index(self, classifier_type: str, metadata: ClassifierStorageMetadata) -> None:
        """Update index.json for a classifier type."""
        index_path = self.classifiers_dir / classifier_type / "index.json"
        
        # Load existing index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {
                'classifier_type': classifier_type,
                'files': [],
                'updated_at': datetime.now().isoformat()
            }
        
        # Add new entry
        index['files'].append(asdict(metadata))
        index['updated_at'] = datetime.now().isoformat()
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def load_signal_data(self, file_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Load signal data from Parquet file."""
        return self.signal_storage.from_parquet(Path(file_path))
    
    def load_classifier_data(self, file_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Load classifier data from Parquet file."""
        return self.classifier_storage.from_parquet(Path(file_path))
    
    def list_signals(self, strategy_type: Optional[str] = None) -> List[SignalStorageMetadata]:
        """List all stored signals, optionally filtered by type."""
        all_signals = []
        
        if strategy_type:
            # List specific type
            type_dirs = [self.signals_dir / strategy_type]
        else:
            # List all types
            type_dirs = [d for d in self.signals_dir.iterdir() if d.is_dir()]
        
        for type_dir in type_dirs:
            index_path = type_dir / "index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                    for file_meta in index['files']:
                        all_signals.append(SignalStorageMetadata(**file_meta))
        
        return all_signals
    
    def list_classifiers(self, classifier_type: Optional[str] = None) -> List[ClassifierStorageMetadata]:
        """List all stored classifiers, optionally filtered by type."""
        all_classifiers = []
        
        if classifier_type:
            # List specific type
            type_dirs = [self.classifiers_dir / classifier_type]
        else:
            # List all types
            type_dirs = [d for d in self.classifiers_dir.iterdir() if d.is_dir()]
        
        for type_dir in type_dirs:
            index_path = type_dir / "index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                    for file_meta in index['files']:
                        all_classifiers.append(ClassifierStorageMetadata(**file_meta))
        
        return all_classifiers
    
    def migrate_from_json(self, json_file_path: str) -> Optional[SignalStorageMetadata]:
        """
        Migrate a JSON signal file to hierarchical Parquet storage.
        
        Args:
            json_file_path: Path to JSON file with signal data
            
        Returns:
            Storage metadata if successful, None otherwise
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            metadata = data.get('metadata', {})
            changes = data.get('changes', [])
            
            # Extract strategy info
            strategies = metadata.get('strategies', {})
            if not strategies:
                logger.warning(f"No strategy info in {json_file_path}")
                return None
            
            # Get first strategy (usually only one per file)
            strategy_key = list(strategies.keys())[0]
            strategy_info = strategies[strategy_key]
            
            # Extract strategy name and parameters
            strategy_name = strategy_info.get('strategy_id', strategy_key).split('_', 1)[-1]
            
            # Try to get parameters from various sources
            parameters = {}
            if 'strategy_parameters' in metadata:
                parameters = metadata['strategy_parameters']
            elif 'metadata' in strategy_info and 'params' in strategy_info['metadata']:
                parameters = strategy_info['metadata']['params']
            
            # Store in hierarchical structure
            return self.store_signal_data(
                signal_changes=changes,
                strategy_name=strategy_name,
                parameters=parameters,
                metadata={
                    'total_bars': metadata.get('total_bars', 0),
                    'symbol': strategy_key.split('_')[0],
                    'timeframe': '1m',  # Default
                    **metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to migrate {json_file_path}: {e}")
            return None