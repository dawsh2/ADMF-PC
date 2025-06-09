"""
Signal storage components using Protocol + Composition.
No inheritance, just protocols and concrete implementations.

This module provides sparse storage for signals and classifier states,
enabling efficient replay of trading strategies without recomputation.
"""

from typing import Protocol, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


# Protocols
class SignalStorageProtocol(Protocol):
    """Protocol for signal storage backends."""
    def append(self, data: Any) -> None: ...
    def save(self, filepath: Path) -> None: ...
    def load(self, filepath: Path) -> None: ...
    def query(self, start_idx: int, end_idx: int) -> List[Any]: ...


class ClassifierStateProtocol(Protocol):
    """Protocol for classifier state tracking."""
    def get_state_at_bar(self, bar_idx: int) -> Optional[str]: ...
    def record_change(self, bar_idx: int, old_state: str, new_state: str) -> None: ...


# Concrete implementations
@dataclass
class ClassifierChangeIndex:
    """Tracks classifier state changes sparsely."""
    classifier_name: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_change(self, bar_idx: int, old_regime: str, new_regime: str) -> None:
        """Record a regime change."""
        self.changes.append({
            'bar_idx': bar_idx,
            'old': old_regime,
            'new': new_regime,
            'timestamp': datetime.now().isoformat()
        })
        logger.debug(f"Classifier {self.classifier_name} changed from {old_regime} to {new_regime} at bar {bar_idx}")
    
    def get_state_at_bar(self, bar_idx: int) -> Optional[str]:
        """Get classifier state at specific bar."""
        state = None
        for change in self.changes:
            if change['bar_idx'] <= bar_idx:
                state = change['new']
            else:
                break
        return state
    
    def save(self, filepath: Path) -> None:
        """Save to Parquet format."""
        if self.changes:
            df = pd.DataFrame(self.changes)
            df.to_parquet(filepath, compression='snappy')
            logger.info(f"Saved {len(self.changes)} classifier changes to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """Load from Parquet format."""
        if filepath.exists():
            df = pd.read_parquet(filepath)
            self.changes = df.to_dict('records')
            logger.info(f"Loaded {len(self.changes)} classifier changes from {filepath}")


@dataclass
class SignalIndex:
    """Stores signals sparsely with strategy parameters."""
    strategy_name: str
    strategy_id: str  # Unique identifier for this parameter set
    parameters: Dict[str, Any]  # The actual parameters used
    signals: List[Dict[str, Any]] = field(default_factory=list)
    
    def append_signal(self, bar_idx: int, signal_value: float, 
                     symbol: str, timeframe: str,
                     classifier_states: Optional[Dict[str, str]] = None,
                     bar_data: Optional[Dict[str, Any]] = None) -> None:
        """Append a signal with full context."""
        entry = {
            'bar_idx': bar_idx,
            'symbol': symbol,
            'timeframe': timeframe,
            'value': signal_value,
            'strategy_id': self.strategy_id,
            'timestamp': datetime.now().isoformat()
        }
        
        if classifier_states:
            entry['classifiers'] = classifier_states
        
        if bar_data:
            # Store minimal bar data for replay
            entry['bar_data'] = {
                'open': bar_data.get('open'),
                'high': bar_data.get('high'),
                'low': bar_data.get('low'),
                'close': bar_data.get('close'),
                'volume': bar_data.get('volume')
            }
        
        self.signals.append(entry)
    
    def get_signals_at_bar(self, bar_idx: int) -> List[Dict[str, Any]]:
        """Get all signals at a specific bar index."""
        return [s for s in self.signals if s['bar_idx'] == bar_idx]
    
    def get_signals_in_range(self, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Get signals within a bar index range."""
        return [s for s in self.signals if start_idx <= s['bar_idx'] <= end_idx]
    
    def save(self, filepath: Path) -> None:
        """Save signals and metadata."""
        # Save metadata
        metadata = {
            'strategy_name': self.strategy_name,
            'strategy_id': self.strategy_id,
            'parameters': self.parameters,
            'signal_count': len(self.signals),
            'created_at': datetime.now().isoformat()
        }
        
        meta_path = filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save signals
        if self.signals:
            df = pd.DataFrame(self.signals)
            df.to_parquet(filepath, compression='snappy')
            logger.info(f"Saved {len(self.signals)} signals to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """Load signals and metadata."""
        # Load metadata
        meta_path = filepath.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
                self.strategy_name = metadata['strategy_name']
                self.strategy_id = metadata['strategy_id']
                self.parameters = metadata['parameters']
        
        # Load signals
        if filepath.exists():
            df = pd.read_parquet(filepath)
            self.signals = df.to_dict('records')
            logger.info(f"Loaded {len(self.signals)} signals from {filepath}")


@dataclass
class MultiSymbolSignal:
    """Signal referencing multiple symbols/timeframes."""
    primary_symbol: str
    primary_timeframe: str
    primary_bar_idx: int
    symbol_refs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    signal_value: float
    strategy_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_symbol_reference(self, symbol: str, timeframe: str, bar_idx: int) -> None:
        """Add a reference to another symbol/timeframe."""
        self.symbol_refs[symbol] = {'timeframe': timeframe, 'bar_idx': bar_idx}
    
    def to_sparse_format(self) -> Dict[str, Any]:
        """Convert to sparse storage format."""
        return {
            'p': f"{self.primary_symbol}|{self.primary_timeframe}|{self.primary_bar_idx}",
            'refs': [(s, d['timeframe'], d['bar_idx']) for s, d in self.symbol_refs.items()],
            'v': self.signal_value,
            'sid': self.strategy_id,
            'm': self.metadata
        }
    
    @classmethod
    def from_sparse_format(cls, data: Dict[str, Any]) -> 'MultiSymbolSignal':
        """Create from sparse format."""
        # Parse primary reference
        primary_parts = data['p'].split('|')
        instance = cls(
            primary_symbol=primary_parts[0],
            primary_timeframe=primary_parts[1],
            primary_bar_idx=int(primary_parts[2]),
            signal_value=data['v'],
            strategy_id=data['sid'],
            metadata=data.get('m', {})
        )
        
        # Add symbol references
        for symbol, timeframe, bar_idx in data.get('refs', []):
            instance.add_symbol_reference(symbol, timeframe, bar_idx)
        
        return instance


@dataclass
class SignalStorageManager:
    """Manages signal and classifier storage for a workflow."""
    base_path: Path
    workflow_id: str
    
    # Storage indices
    signal_indices: Dict[str, SignalIndex] = field(default_factory=dict)
    classifier_indices: Dict[str, ClassifierChangeIndex] = field(default_factory=dict)
    
    def __post_init__(self):
        """Create directory structure."""
        self.signals_dir = self.base_path / self.workflow_id / 'signals'
        self.classifiers_dir = self.base_path / self.workflow_id / 'classifier_changes'
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.classifiers_dir.mkdir(parents=True, exist_ok=True)
    
    def get_or_create_signal_index(self, strategy_id: str, strategy_name: str, 
                                  parameters: Dict[str, Any]) -> SignalIndex:
        """Get existing or create new signal index."""
        if strategy_id not in self.signal_indices:
            self.signal_indices[strategy_id] = SignalIndex(
                strategy_name=strategy_name,
                strategy_id=strategy_id,
                parameters=parameters
            )
        return self.signal_indices[strategy_id]
    
    def get_or_create_classifier_index(self, classifier_name: str) -> ClassifierChangeIndex:
        """Get existing or create new classifier index."""
        if classifier_name not in self.classifier_indices:
            self.classifier_indices[classifier_name] = ClassifierChangeIndex(classifier_name)
        return self.classifier_indices[classifier_name]
    
    def save_all(self) -> None:
        """Save all indices to disk."""
        # Save signal indices
        for strategy_id, index in self.signal_indices.items():
            filepath = self.signals_dir / f"{strategy_id}.parquet"
            index.save(filepath)
        
        # Save classifier indices
        for classifier_name, index in self.classifier_indices.items():
            filepath = self.classifiers_dir / f"{classifier_name}.parquet"
            index.save(filepath)
        
        logger.info(f"Saved {len(self.signal_indices)} signal indices and "
                   f"{len(self.classifier_indices)} classifier indices")
    
    def load_all(self) -> None:
        """Load all indices from disk."""
        # Load signal indices
        for signal_file in self.signals_dir.glob("*.parquet"):
            strategy_id = signal_file.stem
            index = SignalIndex("", strategy_id, {})
            index.load(signal_file)
            self.signal_indices[strategy_id] = index
        
        # Load classifier indices
        for clf_file in self.classifiers_dir.glob("*.parquet"):
            classifier_name = clf_file.stem
            index = ClassifierChangeIndex(classifier_name)
            index.load(clf_file)
            self.classifier_indices[classifier_name] = index
        
        logger.info(f"Loaded {len(self.signal_indices)} signal indices and "
                   f"{len(self.classifier_indices)} classifier indices")