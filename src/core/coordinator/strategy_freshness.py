"""
Strategy Freshness Checker

Determines if strategies need re-computation based on:
1. New data availability
2. Missing trace files  
3. Changed parameters
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyFreshnessChecker:
    """
    Checks if strategy traces are up-to-date with available data.
    
    This enables intelligent re-running where only strategies with new data
    or changed parameters need recomputation.
    """
    
    def __init__(self, traces_dir: str = "./traces", data_dir: str = "./data"):
        self.traces_dir = Path(traces_dir)
        self.data_dir = Path(data_dir)
    
    def check_freshness(
        self, 
        symbols: List[str], 
        strategy_configs: List[Dict[str, Any]],
        data_end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check which strategies need recomputation.
        
        Args:
            symbols: List of symbols to check
            strategy_configs: List of strategy configurations
            data_end_date: Optional end date of available data
            
        Returns:
            Dictionary with freshness status and recommendations
        """
        results = {
            'all_fresh': True,
            'strategies_needing_update': [],
            'reasons': {},
            'trace_metadata': {}
        }
        
        # Import trace store for checking existing traces
        try:
            from ...core.events.tracing.trace_store import TraceStore
            trace_store = TraceStore(str(self.traces_dir))
        except ImportError:
            logger.warning("TraceStore not available - assuming all strategies need update")
            results['all_fresh'] = False
            results['strategies_needing_update'] = [s.get('name', s.get('type')) for s in strategy_configs]
            return results
        
        # Check each strategy
        for strategy_config in strategy_configs:
            strategy_type = strategy_config.get('type')
            strategy_name = strategy_config.get('name', strategy_type)
            
            # Check if trace exists
            trace_path = trace_store.get_trace_path(strategy_type, strategy_name)
            
            if not trace_path.exists():
                # No trace exists - needs computation
                results['all_fresh'] = False
                results['strategies_needing_update'].append(strategy_name)
                results['reasons'][strategy_name] = "No trace file found"
                logger.info(f"Strategy {strategy_name} needs update: no trace file")
                continue
            
            # Load trace metadata
            try:
                metadata_path = trace_path.parent / f"{trace_path.stem}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        results['trace_metadata'][strategy_name] = metadata
                else:
                    # Try to infer from trace file
                    df = pd.read_parquet(trace_path)
                    if not df.empty:
                        metadata = {
                            'start_date': str(df['timestamp'].min()),
                            'end_date': str(df['timestamp'].max()),
                            'total_signals': len(df),
                            'symbols': list(df['symbol'].unique())
                        }
                        results['trace_metadata'][strategy_name] = metadata
                    else:
                        metadata = {}
                
                # Check data freshness
                if data_end_date and 'end_date' in metadata:
                    trace_end = pd.to_datetime(metadata['end_date'])
                    if data_end_date > trace_end:
                        results['all_fresh'] = False
                        results['strategies_needing_update'].append(strategy_name)
                        results['reasons'][strategy_name] = f"New data available after {trace_end}"
                        logger.info(f"Strategy {strategy_name} needs update: new data available")
                        continue
                
                # Check for missing symbols
                if 'symbols' in metadata:
                    trace_symbols = set(metadata['symbols'])
                    requested_symbols = set(symbols)
                    missing_symbols = requested_symbols - trace_symbols
                    
                    if missing_symbols:
                        results['all_fresh'] = False
                        results['strategies_needing_update'].append(strategy_name)
                        results['reasons'][strategy_name] = f"Missing symbols: {missing_symbols}"
                        logger.info(f"Strategy {strategy_name} needs update: missing symbols {missing_symbols}")
                        continue
                
                # Check parameter changes
                if 'strategy_params' in metadata:
                    # Compare parameter hashes
                    current_params = strategy_config.get('param_overrides', {})
                    current_hash = self._compute_param_hash(current_params)
                    stored_hash = metadata.get('param_hash', '')
                    
                    if current_hash != stored_hash:
                        results['all_fresh'] = False
                        results['strategies_needing_update'].append(strategy_name)
                        results['reasons'][strategy_name] = "Parameters changed"
                        logger.info(f"Strategy {strategy_name} needs update: parameters changed")
                        continue
                
                # Strategy is fresh
                logger.info(f"Strategy {strategy_name} is up-to-date")
                
            except Exception as e:
                logger.error(f"Error checking freshness for {strategy_name}: {e}")
                results['all_fresh'] = False
                results['strategies_needing_update'].append(strategy_name)
                results['reasons'][strategy_name] = f"Error checking trace: {e}"
        
        return results
    
    def get_data_end_date(self, symbols: List[str]) -> Optional[datetime]:
        """
        Get the latest available data date across all symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Latest data date or None if no data found
        """
        latest_date = None
        
        for symbol in symbols:
            # Try different file patterns
            patterns = [
                f"{symbol}.csv",
                f"{symbol}_*.csv",
                f"{symbol}/*.csv"
            ]
            
            for pattern in patterns:
                files = list(self.data_dir.glob(pattern))
                for file in files:
                    try:
                        # Quick check - read last few rows
                        df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
                        if not df.empty:
                            file_end_date = df.index[-1]
                            if latest_date is None or file_end_date > latest_date:
                                latest_date = file_end_date
                    except Exception as e:
                        logger.debug(f"Could not read {file}: {e}")
        
        return latest_date
    
    def _compute_param_hash(self, params: Dict[str, Any]) -> str:
        """Compute hash of strategy parameters for change detection."""
        # Sort keys for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
    
    def suggest_update_command(self, strategies_needing_update: List[str], config_path: str) -> str:
        """
        Generate command to update only the strategies that need it.
        
        Args:
            strategies_needing_update: List of strategy names needing update
            config_path: Path to configuration file
            
        Returns:
            Suggested command string
        """
        if not strategies_needing_update:
            return "All strategies are up-to-date!"
        
        # For now, suggest full signal generation
        # Future: could generate partial config with only needed strategies
        return f"python main.py --config {config_path} --signal-generation"


def check_strategy_freshness(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to check strategy freshness from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Freshness check results
    """
    checker = StrategyFreshnessChecker(
        traces_dir=config.get('traces_dir', './traces'),
        data_dir=config.get('data_dir', './data')
    )
    
    # Extract symbols
    symbols = config.get('symbols', [])
    if not isinstance(symbols, list):
        symbols = [symbols]
    
    # Extract strategy configs
    strategy_configs = []
    if 'strategies' in config:
        strategy_configs = config['strategies']
    elif 'parameter_space' in config and 'strategies' in config['parameter_space']:
        strategy_configs = config['parameter_space']['strategies']
    
    # Get latest data date
    data_end_date = checker.get_data_end_date(symbols)
    
    # Check freshness
    results = checker.check_freshness(symbols, strategy_configs, data_end_date)
    
    # Add suggestions
    if not results['all_fresh']:
        config_path = config.get('metadata', {}).get('config_file', 'config.yaml')
        results['update_command'] = checker.suggest_update_command(
            results['strategies_needing_update'], 
            config_path
        )
    
    return results