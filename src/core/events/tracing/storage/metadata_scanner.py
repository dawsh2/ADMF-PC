# Fast Metadata Scanner for Grid Search Results
"""
Provides fast scanning capabilities for grid search workspaces
without loading the actual signal/classifier data.
"""

from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import json
import pandas as pd
from collections import defaultdict
import pyarrow.parquet as pq


class GridSearchScanner:
    """Fast scanning for grid search workspaces"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self._manifest_cache = {}
        self._index_cache = {}
        
    def scan_all_workspaces(
        self, 
        workflow_type: Optional[str] = None,
        after_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Scan all workspaces with optional filters
        
        Args:
            workflow_type: Filter by workflow type (grid_search, optimization, etc.)
            after_date: Only include runs after this date
            symbols: Filter by symbols
            
        Returns:
            DataFrame with workspace summaries
        """
        results = []
        
        for workspace_dir in self.workspace_root.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            # Quick manifest check
            manifest_path = workspace_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            
            manifest = self._load_manifest(manifest_path)
            
            # Apply filters
            if workflow_type and manifest.get('workflow', {}).get('type') != workflow_type:
                continue
            
            if after_date:
                created_at = datetime.fromisoformat(manifest.get('created_at', ''))
                if created_at < after_date:
                    continue
            
            if symbols:
                workspace_symbols = set(manifest.get('data', {}).get('symbols', []))
                if not workspace_symbols.intersection(symbols):
                    continue
            
            # Extract summary info
            summary = manifest.get('summary', {})
            perf_summary = manifest.get('performance_summary', {})
            
            results.append({
                'workspace': workspace_dir.name,
                'created_at': manifest.get('created_at'),
                'workflow_type': manifest.get('workflow', {}).get('type'),
                'symbols': ', '.join(manifest.get('data', {}).get('symbols', [])),
                'total_strategies': summary.get('total_strategies', 0),
                'total_classifiers': summary.get('total_classifiers', 0),
                'best_sharpe': perf_summary.get('best_strategy', {}).get('sharpe'),
                'best_strategy': perf_summary.get('best_strategy', {}).get('id'),
                'path': str(workspace_dir)
            })
        
        return pd.DataFrame(results)
    
    def scan_performance(
        self,
        workspace_name: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        strategy_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Scan strategy performance across workspaces
        
        Args:
            workspace_name: Specific workspace to scan
            min_sharpe: Minimum Sharpe ratio filter
            max_drawdown: Maximum drawdown filter
            strategy_types: Filter by strategy types
            
        Returns:
            DataFrame with strategy performances
        """
        results = []
        
        # Determine workspaces to scan
        if workspace_name:
            workspaces = [self.workspace_root / workspace_name]
        else:
            workspaces = [d for d in self.workspace_root.iterdir() if d.is_dir()]
        
        for workspace_dir in workspaces:
            if not workspace_dir.exists():
                continue
            
            # Load strategy index
            strategy_index_path = workspace_dir / "signals" / "index.json"
            if not strategy_index_path.exists():
                continue
            
            with open(strategy_index_path) as f:
                index = json.load(f)
            
            # Extract all strategy performances
            for strategy_type, type_data in index.get('strategies', {}).items():
                # Apply strategy type filter
                if strategy_types and strategy_type not in strategy_types:
                    continue
                
                for strategy_id, strategy_info in type_data.get('files', {}).items():
                    perf = strategy_info.get('performance', {})
                    
                    # Apply performance filters
                    if min_sharpe and perf.get('sharpe', 0) < min_sharpe:
                        continue
                    
                    if max_drawdown and abs(perf.get('max_drawdown', -1)) > max_drawdown:
                        continue
                    
                    results.append({
                        'workspace': workspace_dir.name,
                        'strategy_type': strategy_type,
                        'strategy_id': strategy_id,
                        'params': strategy_info.get('params', {}),
                        'sharpe': perf.get('sharpe'),
                        'sortino': perf.get('sortino'),
                        'max_drawdown': perf.get('max_drawdown'),
                        'win_rate': perf.get('win_rate'),
                        'profit_factor': perf.get('profit_factor'),
                        'signal_changes': strategy_info.get('signal_changes'),
                        'compression_ratio': strategy_info.get('compression_ratio'),
                        'file': strategy_info.get('file')
                    })
        
        df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio by default
        if 'sharpe' in df.columns:
            df = df.sort_values('sharpe', ascending=False)
        
        return df
    
    def scan_classifiers(
        self,
        workspace_name: Optional[str] = None,
        min_stability: Optional[float] = None
    ) -> pd.DataFrame:
        """Scan classifier performance
        
        Args:
            workspace_name: Specific workspace to scan
            min_stability: Minimum regime stability score
            
        Returns:
            DataFrame with classifier information
        """
        results = []
        
        # Similar to scan_performance but for classifiers
        if workspace_name:
            workspaces = [self.workspace_root / workspace_name]
        else:
            workspaces = [d for d in self.workspace_root.iterdir() if d.is_dir()]
        
        for workspace_dir in workspaces:
            classifier_index_path = workspace_dir / "classifiers" / "index.json"
            if not classifier_index_path.exists():
                continue
            
            with open(classifier_index_path) as f:
                index = json.load(f)
            
            for classifier_type, type_data in index.get('classifiers', {}).items():
                for classifier_id, classifier_info in type_data.get('files', {}).items():
                    stats = classifier_info.get('statistics', {})
                    
                    if min_stability and stats.get('regime_stability', 0) < min_stability:
                        continue
                    
                    results.append({
                        'workspace': workspace_dir.name,
                        'classifier_type': classifier_type,
                        'classifier_id': classifier_id,
                        'params': classifier_info.get('params', {}),
                        'regime_changes': stats.get('regime_changes'),
                        'avg_regime_duration': stats.get('avg_regime_duration'),
                        'regime_stability': stats.get('regime_stability'),
                        'compression_ratio': classifier_info.get('compression_ratio')
                    })
        
        return pd.DataFrame(results)
    
    def find_best_combinations(
        self,
        workspace_name: str,
        metric: str = 'sharpe',
        top_n: int = 10
    ) -> pd.DataFrame:
        """Find best strategy-classifier combinations
        
        Args:
            workspace_name: Workspace to analyze
            metric: Metric to optimize
            top_n: Number of top combinations to return
            
        Returns:
            DataFrame with top combinations
        """
        workspace_dir = self.workspace_root / workspace_name
        
        # Load analytics if available
        analytics_dir = workspace_dir / 'analytics'
        if analytics_dir.exists():
            # Try to load pre-computed regime performance
            regime_perf_path = analytics_dir / 'regime_performance.parquet'
            if regime_perf_path.exists():
                return pd.read_parquet(regime_perf_path).head(top_n)
        
        # Otherwise compute from indices
        strategies_df = self.scan_performance(workspace_name)
        classifiers_df = self.scan_classifiers(workspace_name)
        
        # Simple cross-product for now
        # In reality, you'd load the actual regime-aware performance
        results = []
        for _, strategy in strategies_df.iterrows():
            for _, classifier in classifiers_df.iterrows():
                # Estimate combined performance
                combined_score = strategy.get(metric, 0) * classifier.get('regime_stability', 1)
                
                results.append({
                    'strategy_id': strategy['strategy_id'],
                    'classifier_id': classifier['classifier_id'],
                    'strategy_type': strategy['strategy_type'],
                    'classifier_type': classifier['classifier_type'],
                    f'estimated_{metric}': combined_score
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values(f'estimated_{metric}', ascending=False)
        
        return df.head(top_n)
    
    def get_parameter_sensitivity(
        self,
        workspace_name: str,
        strategy_type: str,
        parameter: str,
        metric: str = 'sharpe'
    ) -> pd.DataFrame:
        """Analyze parameter sensitivity
        
        Args:
            workspace_name: Workspace to analyze
            strategy_type: Type of strategy
            parameter: Parameter to analyze
            metric: Performance metric
            
        Returns:
            DataFrame with parameter vs performance
        """
        strategies_df = self.scan_performance(
            workspace_name,
            strategy_types=[strategy_type]
        )
        
        # Extract parameter values
        param_values = []
        metric_values = []
        
        for _, row in strategies_df.iterrows():
            params = row.get('params', {})
            if parameter in params:
                param_values.append(params[parameter])
                metric_values.append(row.get(metric, 0))
        
        return pd.DataFrame({
            parameter: param_values,
            metric: metric_values
        }).sort_values(parameter)
    
    def _load_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Load and cache manifest"""
        if manifest_path in self._manifest_cache:
            return self._manifest_cache[manifest_path]
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        self._manifest_cache[manifest_path] = manifest
        return manifest
    
    def clear_cache(self):
        """Clear internal caches"""
        self._manifest_cache.clear()
        self._index_cache.clear()


class PerformanceScanner:
    """Specialized scanner for performance metrics"""
    
    @staticmethod
    def load_performance_matrix(workspace_path: Path) -> Optional[pd.DataFrame]:
        """Load pre-computed performance matrix"""
        matrix_path = workspace_path / 'analytics' / 'performance_matrix.parquet'
        if matrix_path.exists():
            return pd.read_parquet(matrix_path)
        return None
    
    @staticmethod
    def load_correlation_matrix(workspace_path: Path) -> Optional[pd.DataFrame]:
        """Load strategy correlation matrix"""
        corr_path = workspace_path / 'analytics' / 'correlation_matrix.parquet'
        if corr_path.exists():
            return pd.read_parquet(corr_path)
        return None
    
    @staticmethod
    def quick_summary(workspace_path: Path) -> Dict[str, Any]:
        """Get quick performance summary"""
        summary_path = workspace_path / 'analytics' / 'summary_stats.json'
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        
        # Fallback to manifest
        manifest_path = workspace_path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
                return manifest.get('performance_summary', {})
        
        return {}