# Universal Strategy Storage System

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Union
import yaml

class StrategyStorage:
    """
    Universal storage system for all trading strategies.
    Handles simple indicators, composites, conditionals, and any nested structure.
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.traces_dir = results_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate storage for full strategy configurations
        self.strategy_configs_dir = results_dir / "strategy_configs"
        self.strategy_configs_dir.mkdir(exist_ok=True)
        
    def _flatten_strategy(self, strategy_config: Union[Dict, List], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten any strategy configuration into analyzable key-value pairs.
        
        Example input:
        {
            'weight': 0.6,
            'strategy': [
                {'sma_crossover': {'fast': 10, 'slow': 30, 'weight': 0.5}},
                {'momentum': {'period': 14, 'weight': 0.5}}
            ],
            'threshold': "0.3 AND adx(14) > 25"
        }
        
        Output: Flattened representation with paths as keys
        """
        flat = {}
        
        if isinstance(strategy_config, dict):
            for key, value in strategy_config.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    flat.update(self._flatten_strategy(value, new_prefix))
                else:
                    flat[new_prefix] = value
                    
        elif isinstance(strategy_config, list):
            for i, item in enumerate(strategy_config):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                flat.update(self._flatten_strategy(item, new_prefix))
                
        return flat
    
    def _compute_strategy_hash(self, strategy_config: Any) -> str:
        """Compute deterministic hash for any strategy configuration"""
        # Ensure deterministic JSON encoding
        config_str = json.dumps(strategy_config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def save_strategy_trace(self,
                           signals_df: pd.DataFrame,
                           strategy_config: Union[Dict, List],
                           strategy_id: int,
                           symbol: str,
                           timeframe: str,
                           performance_metrics: Dict[str, float] = None) -> Path:
        """
        Save trace with complete strategy configuration for any strategy type.
        """
        
        # Compute strategy hash
        strategy_hash = self._compute_strategy_hash(strategy_config)
        
        # Determine primary strategy type
        strategy_type = self._get_strategy_type(strategy_config)
        
        # Add metadata columns to signals
        signals_df = signals_df.copy()
        signals_df['idx'] = signals_df.index  # Track bar index
        signals_df['strategy_id'] = strategy_id
        signals_df['strategy_hash'] = strategy_hash
        signals_df['strategy_type'] = strategy_type
        signals_df['symbol'] = symbol
        signals_df['timeframe'] = timeframe
        
        # Flatten strategy config and add as columns
        flat_config = self._flatten_strategy(strategy_config)
        for key, value in flat_config.items():
            if isinstance(value, (int, float, str, bool)):
                # Clean key for column name
                clean_key = key.replace('.', '_').replace('[', '_').replace(']', '')
                signals_df[f'meta_{clean_key}'] = value
        
        # Save strategy configuration
        config_file = self.strategy_configs_dir / f"strategy_{strategy_id}.json"
        config_data = {
            'strategy_id': strategy_id,
            'strategy_hash': strategy_hash,
            'strategy_type': strategy_type,
            'symbol': symbol,
            'timeframe': timeframe,
            'config': strategy_config,  # The exact config for this instance
            'flattened': flat_config,   # Flattened version for querying
            'performance_metrics': performance_metrics or {}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Universal filename format
        filename = f"strategy_{strategy_id}.parquet"
        filepath = self.traces_dir / filename
        
        # Add configuration to parquet metadata
        metadata = {
            'strategy_id': str(strategy_id),
            'strategy_hash': strategy_hash,
            'strategy_type': strategy_type,
            'config_file': str(config_file.relative_to(self.results_dir)),
            'strategy_config': json.dumps(strategy_config)  # The exact instance config
        }
        
        table = pa.Table.from_pandas(signals_df)
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, filepath)
        
        return filepath
    
    def _get_strategy_type(self, config: Union[Dict, List]) -> str:
        """Determine the primary strategy type"""
        if isinstance(config, list):
            return "ensemble"
        elif isinstance(config, dict):
            if 'strategy' in config and isinstance(config['strategy'], list):
                return "composite"
            elif 'condition' in config:
                return "conditional"
            else:
                # Find the strategy name (first key that's not a meta-key)
                meta_keys = {'weight', 'threshold', 'risk', 'filter', 'condition', 'timeframe'}
                for key in config:
                    if key not in meta_keys:
                        return key
        return "unknown"
    
    def create_strategy_index(self) -> pd.DataFrame:
        """Create a comprehensive index of all strategies"""
        index_data = []
        
        # Load all strategy configurations
        for config_file in self.strategy_configs_dir.glob("*.json"):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Create index entry
            entry = {
                'strategy_id': config_data['strategy_id'],
                'strategy_hash': config_data['strategy_hash'],
                'strategy_type': config_data['strategy_type'],
                'symbol': config_data['symbol'],
                'timeframe': config_data['timeframe'],
                'config_file': str(config_file.name)
            }
            
            # Add flattened parameters for filtering
            for flat_key, flat_value in config_data['flattened'].items():
                if isinstance(flat_value, (int, float, str, bool)):
                    # Clean up the key for better column names
                    clean_key = flat_key.replace('[', '_').replace(']', '').replace('.', '_')
                    entry[f'flat_{clean_key}'] = flat_value
            
            # Add performance metrics if available
            for metric, value in config_data.get('performance_metrics', {}).items():
                entry[f'metric_{metric}'] = value
                
            index_data.append(entry)
        
        # Create DataFrame and save as parquet for fast queries
        index_df = pd.DataFrame(index_data)
        index_df.to_parquet(self.results_dir / "strategy_index.parquet")
        
        return index_df
    
    def load_strategy_config(self, strategy_id: int = None, strategy_hash: str = None) -> Dict:
        """Load full configuration for a specific strategy"""
        if strategy_id is not None:
            pattern = f"strategy_{strategy_id}.json"
        elif strategy_hash is not None:
            pattern = f"*_{strategy_hash}.json"
        else:
            raise ValueError("Must provide either strategy_id or strategy_hash")
            
        config_files = list(self.strategy_configs_dir.glob(pattern))
        if not config_files:
            raise FileNotFoundError(f"No config found for {pattern}")
            
        with open(config_files[0], 'r') as f:
            return json.load(f)
    
    def query_strategies(self, con, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Query strategies with flexible filters.
        
        Example:
        filters = {
            'strategy_type': 'bollinger_bands',
            'flat_params_period': [20, 25],  # Will be IN (20, 25)
            'metric_sharpe_ratio': (1.0, None)  # Will be >= 1.0
        }
        """
        # Load strategy index
        if not (self.results_dir / "strategy_index.parquet").exists():
            self.create_strategy_index()
            
        query = f"""
        SELECT * FROM read_parquet('{self.results_dir}/strategy_index.parquet')
        """
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # IN clause
                    values_str = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                    conditions.append(f"{key} IN ({values_str})")
                elif isinstance(value, tuple) and len(value) == 2:
                    # Range query
                    if value[0] is not None:
                        conditions.append(f"{key} >= {value[0]}")
                    if value[1] is not None:
                        conditions.append(f"{key} <= {value[1]}")
                else:
                    # Exact match
                    if isinstance(value, str):
                        conditions.append(f"{key} = '{value}'")
                    else:
                        conditions.append(f"{key} = {value}")
                        
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
        return con.execute(query).df()


# Example usage:

def example_usage():
    storage = StrategyStorage(Path("results/run_20241223"))
    
    # Example 1: Simple strategy
    simple_config = {
        'bollinger_bands': {
            'period': 20,
            'std_dev': 2.0
        },
        'threshold': "volume > sma(volume, 20) * 1.2"
    }
    
    # Example 2: Composite strategy
    composite_config = {
        'weight': 0.6,
        'strategy': [
            {'ma_crossover': {'fast': 10, 'slow': 30, 'weight': 0.5}},
            {'momentum': {'period': 14, 'weight': 0.5}}
        ],
        'threshold': "0.3 AND adx(14) > 25"
    }
    
    # Example 3: Complex nested strategy
    complex_config = [
        {
            'condition': "volatility_regime(20) == 'high'",
            'weight': 0.4,
            'strategy': [
                {'bollinger_breakout': {'weight': 0.6, 'params': {'period': 20}}},
                {'keltner_breakout': {'weight': 0.4, 'params': {'period': 20}}}
            ]
        },
        {
            'condition': "volatility_regime(20) == 'low'",
            'weight': 0.4,
            'mean_reversion': {'period': 20, 'threshold': 2.0}
        },
        {
            'weight': 0.2,
            'vwap_deviation': {'std_multiplier': 2.0}
        }
    ]
    
    # All use the same storage method
    signals_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'signal': [0, 1, 1, 0, -1] * 20
    })
    
    storage.save_strategy_trace(signals_df, simple_config, 0, "SPY", "5m")
    storage.save_strategy_trace(signals_df, composite_config, 1, "SPY", "5m")
    storage.save_strategy_trace(signals_df, complex_config, 2, "SPY", "5m")
    
    # Query strategies
    con = duckdb.connect()
    
    # Find all bollinger strategies
    bb_strategies = storage.query_strategies(con, {'strategy_type': 'bollinger_bands'})
    
    # Find strategies with good performance
    good_performers = storage.query_strategies(con, {
        'metric_sharpe_ratio': (1.0, None)
    })
    
    # Load full config for analysis
    config = storage.load_strategy_config(strategy_id=2)
    print(json.dumps(config, indent=2))
