"""
Simple Parquet Storage for Hierarchical Structure

A minimal implementation that creates the hierarchical storage structure
without depending on the analytics module.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StorageMetadata:
    """Generic storage metadata"""
    type: str
    name: str
    parameters: Dict[str, Any]
    total_items: int
    compression_ratio: float
    created_at: str
    file_hash: str
    file_path: str


class MinimalHierarchicalStorage:
    """
    Minimal hierarchical storage that creates clean directory structure.
    Stores signal changes as JSON for now (can be upgraded to Parquet later).
    """
    
    def __init__(self, base_dir: str = "./analytics_storage"):
        """Initialize storage manager with base directory."""
        self.base_dir = Path(base_dir)
        self.signals_dir = self.base_dir / "signals"
        self.classifiers_dir = self.base_dir / "classifiers"
        
        # Create base directories
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.classifiers_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized minimal hierarchical storage at {self.base_dir}")
    
    def _generate_param_hash(self, params: Dict[str, Any]) -> str:
        """Generate a hash from parameters for unique filename."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()[:8]
    
    def _get_strategy_type(self, strategy_name: str) -> str:
        """Extract strategy type from name."""
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
    
    def _get_classifier_type(self, classifier_name: str) -> str:
        """Extract classifier type from name."""
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
    
    def _create_filename(self, name: str, params: Dict[str, Any], extension: str = 'json') -> str:
        """Create filename for storage."""
        # Extract key parameters for readable filename
        param_parts = []
        
        # Common parameters
        for key in ['fast_period', 'slow_period', 'period', 'lookback_period', 
                    'threshold', 'momentum_threshold', 'n_states']:
            if key in params:
                value = params[key]
                # Shorten key names
                short_key = key.replace('_period', '').replace('lookback_', 'lb')
                param_parts.append(f"{short_key}_{value}")
        
        # Generate hash for uniqueness
        param_hash = self._generate_param_hash(params)
        
        # Build filename
        name_parts = name.split('_')
        base_name = '_'.join(name_parts[:2]) if len(name_parts) > 2 else name
        
        if param_parts:
            filename = f"{base_name}_{'_'.join(param_parts[:3])}_{param_hash}.{extension}"
        else:
            filename = f"{base_name}_{param_hash}.{extension}"
            
        return filename
    
    def store_signal_data(
        self,
        signal_changes: List[Dict[str, Any]],
        strategy_name: str,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> StorageMetadata:
        """Store signal data in hierarchical structure."""
        if not parameters:
            parameters = {}
        if not metadata:
            metadata = {}
            
        # Determine strategy type and create directory
        strat_type = self._get_strategy_type(strategy_name)
        type_dir = self.signals_dir / strat_type
        type_dir.mkdir(exist_ok=True)
        
        # Create filename (using JSON for now)
        filename = self._create_filename(strategy_name, parameters, 'json')
        file_path = type_dir / filename
        
        # Extract metadata
        total_bars = metadata.get('total_bars', 0)
        compression_ratio = len(signal_changes) / total_bars if total_bars > 0 else 0
        
        # Create storage data
        storage_data = {
            'metadata': {
                'strategy_name': strategy_name,
                'strategy_type': strat_type,
                'parameters': parameters,
                'total_bars': total_bars,
                'signal_changes': len(signal_changes),
                'compression_ratio': compression_ratio,
                'created_at': datetime.now().isoformat(),
                **metadata
            },
            'changes': signal_changes
        }
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        # Create metadata object
        storage_meta = StorageMetadata(
            type=strat_type,
            name=strategy_name,
            parameters=parameters,
            total_items=len(signal_changes),
            compression_ratio=compression_ratio,
            created_at=storage_data['metadata']['created_at'],
            file_hash=self._generate_param_hash(parameters),
            file_path=str(file_path)
        )
        
        # Update index
        self._update_index(self.signals_dir / strat_type, storage_meta)
        
        logger.info(f"Stored {len(signal_changes)} signal changes to {file_path}")
        logger.info(f"Compression: {compression_ratio:.2%} ({len(signal_changes)}/{total_bars} bars)")
        
        return storage_meta
    
    def store_classifier_data(
        self,
        regime_changes: List[Dict[str, Any]],
        classifier_name: str,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> StorageMetadata:
        """Store classifier data in hierarchical structure."""
        if not parameters:
            parameters = {}
        if not metadata:
            metadata = {}
            
        # Determine classifier type and create directory
        class_type = self._get_classifier_type(classifier_name)
        type_dir = self.classifiers_dir / class_type
        type_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = self._create_filename(classifier_name, parameters, 'json')
        file_path = type_dir / filename
        
        # Extract metadata
        total_bars = metadata.get('total_bars', 0)
        compression_ratio = len(regime_changes) / total_bars if total_bars > 0 else 0
        
        # Create storage data
        storage_data = {
            'metadata': {
                'classifier_name': classifier_name,
                'classifier_type': class_type,
                'parameters': parameters,
                'total_bars': total_bars,
                'regime_changes': len(regime_changes),
                'compression_ratio': compression_ratio,
                'created_at': datetime.now().isoformat(),
                **metadata
            },
            'changes': regime_changes
        }
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        # Create metadata object
        storage_meta = StorageMetadata(
            type=class_type,
            name=classifier_name,
            parameters=parameters,
            total_items=len(regime_changes),
            compression_ratio=compression_ratio,
            created_at=storage_data['metadata']['created_at'],
            file_hash=self._generate_param_hash(parameters),
            file_path=str(file_path)
        )
        
        # Update index
        self._update_index(self.classifiers_dir / class_type, storage_meta)
        
        logger.info(f"Stored {len(regime_changes)} regime changes to {file_path}")
        logger.info(f"Compression: {compression_ratio:.2%}")
        
        return storage_meta
    
    def _update_index(self, type_dir: Path, metadata: StorageMetadata) -> None:
        """Update index.json for a type directory."""
        index_path = type_dir / "index.json"
        
        # Load existing index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {
                'type': metadata.type,
                'files': [],
                'updated_at': datetime.now().isoformat()
            }
        
        # Add new entry
        index['files'].append(asdict(metadata))
        index['updated_at'] = datetime.now().isoformat()
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)