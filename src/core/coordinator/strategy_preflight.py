"""
Strategy Pre-flight Checker

Checks which strategies need computation before any data processing begins.
This enables smart skipping of already-computed strategies and their features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import hashlib
import json

from ..events.observers.strategy_metadata_extractor import compute_strategy_hash

logger = logging.getLogger(__name__)


class StrategyPreflightChecker:
    """
    Check which strategies need computation before any data processing.
    
    This allows the system to:
    1. Skip data loading if all strategies exist
    2. Only compute features for strategies that need to run
    3. Provide clear feedback about what will be computed
    """
    
    def __init__(self, traces_dir: str = "./traces"):
        self.traces_dir = Path(traces_dir)
        self._existing_hashes = self._load_existing_hashes()
    
    def _load_existing_hashes(self) -> Set[str]:
        """Load existing strategy hashes from global traces directory."""
        existing_hashes = set()
        
        # Check strategy index first (fastest)
        strategy_index_path = self.traces_dir / 'strategy_index.parquet'
        if strategy_index_path.exists():
            try:
                import pandas as pd
                index_df = pd.read_parquet(strategy_index_path)
                existing_hashes.update(index_df['strategy_hash'].unique())
                logger.info(f"Loaded {len(existing_hashes)} existing strategy hashes from index")
                return existing_hashes
            except Exception as e:
                logger.warning(f"Could not load strategy index: {e}")
        
        # Fallback: scan directory structure
        logger.info("Scanning traces directory for existing strategies...")
        for symbol_dir in self.traces_dir.glob("*"):
            if not symbol_dir.is_dir():
                continue
            for timeframe_dir in symbol_dir.glob("*"):
                if not timeframe_dir.is_dir():
                    continue
                signals_dir = timeframe_dir / "signals"
                if signals_dir.exists():
                    for strategy_type_dir in signals_dir.glob("*"):
                        if strategy_type_dir.is_dir():
                            # Hash-based filenames
                            for parquet_file in strategy_type_dir.glob("*.parquet"):
                                if not parquet_file.name.startswith("_"):
                                    # Extract hash from filename
                                    strategy_hash = parquet_file.stem
                                    existing_hashes.add(strategy_hash)
        
        logger.info(f"Found {len(existing_hashes)} existing strategies in traces")
        return existing_hashes
    
    def check_strategies(
        self, 
        strategy_configs: List[Dict[str, Any]], 
        symbol: str = None,
        timeframe: str = None
    ) -> Dict[str, Any]:
        """
        Check which strategies need computation.
        
        Args:
            strategy_configs: List of strategy configurations
            symbol: Optional symbol to check (for more specific checks)
            timeframe: Optional timeframe to check
            
        Returns:
            Dictionary with:
            - all_exist: Boolean if all strategies already exist
            - strategies_to_compute: List of configs that need computation
            - strategies_to_skip: List of configs that can be skipped
            - required_features: Set of features needed by strategies to compute
            - skipped_count: Number of strategies being skipped
            - compute_count: Number of strategies to compute
            - summary: Human-readable summary
        """
        strategies_to_compute = []
        strategies_to_skip = []
        required_features = set()
        strategy_details = []
        
        for config in strategy_configs:
            # Compute hash for this strategy
            strategy_hash = compute_strategy_hash(config)
            strategy_type = config.get('type', 'unknown')
            
            # Check if it exists
            exists = strategy_hash in self._existing_hashes
            
            detail = {
                'type': strategy_type,
                'hash': strategy_hash,
                'exists': exists,
                'parameters': config.get('parameters', {})
            }
            strategy_details.append(detail)
            
            if exists:
                strategies_to_skip.append(config)
                logger.debug(f"Strategy {strategy_type} ({strategy_hash}) already exists")
            else:
                strategies_to_compute.append(config)
                logger.debug(f"Strategy {strategy_type} ({strategy_hash}) needs computation")
                
                # Extract required features for this strategy
                features = self._extract_required_features(config)
                required_features.update(features)
        
        # Prepare result
        result = {
            'all_exist': len(strategies_to_compute) == 0,
            'strategies_to_compute': strategies_to_compute,
            'strategies_to_skip': strategies_to_skip,
            'required_features': required_features,
            'skipped_count': len(strategies_to_skip),
            'compute_count': len(strategies_to_compute),
            'total_count': len(strategy_configs),
            'details': strategy_details
        }
        
        # Add human-readable summary
        if result['all_exist']:
            result['summary'] = f"All {result['total_count']} strategies already computed. Nothing to do."
        else:
            result['summary'] = (
                f"Need to compute {result['compute_count']} of {result['total_count']} strategies "
                f"(skipping {result['skipped_count']} existing)"
            )
        
        return result
    
    def _extract_required_features(self, strategy_config: Dict[str, Any]) -> Set[str]:
        """
        Extract features required by a strategy configuration.
        
        This is a simplified version - in practice, each strategy type
        would declare its required features.
        """
        required = set()
        strategy_type = strategy_config.get('type', '')
        params = strategy_config.get('parameters', {})
        
        # Map common strategy types to their feature requirements
        if 'bollinger' in strategy_type:
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            # Bollinger strategies need bollinger bands features
            required.add(f'bollinger_bands_{period}_{std_dev}_upper')
            required.add(f'bollinger_bands_{period}_{std_dev}_middle')
            required.add(f'bollinger_bands_{period}_{std_dev}_lower')
            
        elif 'rsi' in strategy_type:
            period = params.get('period', 14)
            required.add(f'rsi_{period}')
            
        elif 'ma_crossover' in strategy_type or 'sma' in strategy_type:
            fast = params.get('fast_period', 10)
            slow = params.get('slow_period', 20)
            required.add(f'sma_{fast}')
            required.add(f'sma_{slow}')
            
        elif 'momentum' in strategy_type:
            period = params.get('period', 10)
            required.add(f'momentum_{period}')
            required.add(f'roc_{period}')
            
        elif 'macd' in strategy_type:
            required.add('macd')
            required.add('macd_signal')
            required.add('macd_histogram')
            
        elif 'breakout' in strategy_type:
            lookback = params.get('lookback', 20)
            required.add(f'highest_{lookback}')
            required.add(f'lowest_{lookback}')
            required.add('atr_14')  # Often used for breakout confirmation
            
        # Add any explicitly declared features
        if 'required_features' in strategy_config:
            required.update(strategy_config['required_features'])
        
        return required
    
    def get_required_features_for_configs(
        self, 
        strategy_configs: List[Dict[str, Any]]
    ) -> Set[str]:
        """
        Get all features required by a list of strategy configurations.
        
        Args:
            strategy_configs: List of strategy configurations
            
        Returns:
            Set of all required feature names
        """
        all_features = set()
        for config in strategy_configs:
            features = self._extract_required_features(config)
            all_features.update(features)
        return all_features
    
    def estimate_time_saved(
        self, 
        skipped_count: int, 
        total_bars: int = 16000,
        ms_per_strategy_per_bar: float = 0.5
    ) -> Dict[str, Any]:
        """
        Estimate time saved by skipping existing strategies.
        
        Args:
            skipped_count: Number of strategies being skipped
            total_bars: Total number of bars to process
            ms_per_strategy_per_bar: Estimated milliseconds per strategy per bar
            
        Returns:
            Dictionary with time estimates
        """
        total_operations_skipped = skipped_count * total_bars
        time_saved_ms = total_operations_skipped * ms_per_strategy_per_bar
        time_saved_seconds = time_saved_ms / 1000
        
        return {
            'operations_skipped': total_operations_skipped,
            'time_saved_seconds': time_saved_seconds,
            'time_saved_minutes': time_saved_seconds / 60,
            'time_saved_formatted': self._format_time(time_saved_seconds)
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"