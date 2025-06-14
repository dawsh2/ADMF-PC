# Analytics Integration for ADMF-PC
"""
Automatically creates SQL analytics workspaces when topology runs complete.
Integrates with the coordinator to capture strategy and classifier results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import hashlib
import uuid

from .workspace import AnalyticsWorkspace
from .migration import setup_workspace
from .exceptions import AnalyticsError

logger = logging.getLogger(__name__)


class AnalyticsIntegrator:
    """Integrates SQL analytics with ADMF-PC workflow execution"""
    
    def __init__(self, workspaces_dir: str = "workspaces", auto_create: bool = True):
        """Initialize analytics integrator
        
        Args:
            workspaces_dir: Directory to store analytics workspaces
            auto_create: Whether to automatically create workspaces on run completion
        """
        self.workspaces_dir = Path(workspaces_dir)
        self.auto_create = auto_create
        self.workspaces_dir.mkdir(exist_ok=True)
        
    def create_workspace_for_run(self, 
                                result: Dict[str, Any], 
                                topology_name: str, 
                                config: Dict[str, Any]) -> Optional[Path]:
        """Create SQL analytics workspace for a completed run
        
        Args:
            result: Topology execution result
            topology_name: Name of the topology that was executed
            config: Original configuration used for the run
            
        Returns:
            Path to analytics.duckdb if created successfully, None otherwise
        """
        if not self.auto_create or not result.get('success'):
            return None
            
        try:
            # Try to use existing signal trace workspace if available
            workspace_path = self._get_existing_workspace_path(result, config)
            if workspace_path:
                logger.info(f"Using existing signal workspace: {workspace_path}")
            else:
                # Fallback: Generate new workspace name
                workspace_name = self._generate_workspace_name(result, topology_name, config)
                workspace_path = self.workspaces_dir / workspace_name
                logger.info(f"Creating new analytics workspace: {workspace_name}")
            
            logger.info(f"Setting up SQL analytics in: {workspace_path}")
            
            # Create analytics.duckdb in the workspace directory
            workspace = setup_workspace(workspace_path)
            
            # Extract and insert run data
            run_id = self._insert_run_record(workspace, result, topology_name, config)
            
            # Extract strategies and classifiers from result
            strategies_data = self._extract_strategies_data(result, config)
            classifiers_data = self._extract_classifiers_data(result, config)
            
            # Insert strategy records
            strategy_count = 0
            for strategy_data in strategies_data:
                strategy_data['run_id'] = run_id
                self._insert_strategy_record(workspace, strategy_data)
                strategy_count += 1
            
            # Insert classifier records
            classifier_count = 0
            for classifier_data in classifiers_data:
                classifier_data['run_id'] = run_id
                self._insert_classifier_record(workspace, classifier_data)
                classifier_count += 1
            
            logger.info(f"Analytics database populated: {strategy_count} strategies, "
                       f"{classifier_count} classifiers stored in {workspace_path.name}")
            
            # Update run record with final counts
            workspace.conn.execute("""
                UPDATE runs SET 
                    total_strategies = ?, 
                    total_classifiers = ?,
                    total_combinations = ?
                WHERE run_id = ?
            """, [strategy_count, classifier_count, strategy_count * classifier_count, run_id])
            
            workspace.close()
            return workspace_path / "analytics.duckdb"
            
        except Exception as e:
            logger.error(f"Failed to create analytics workspace: {e}")
            return None

    def _get_existing_workspace_path(self, result: Dict[str, Any], config: Dict[str, Any]) -> Optional[Path]:
        """Extract existing signal workspace path from tracer results"""
        tracer_results = result.get('tracer_results')
        if not tracer_results:
            return None
            
        # Check if workspace path is directly provided by MultiStrategyTracer
        workspace_path = tracer_results.get('workspace_path')
        if workspace_path:
            return Path(workspace_path)
            
        # Fallback: Check if we have signal trace components with file paths
        components = tracer_results.get('components', {})
        if not components:
            return None
            
        # Get any component's file path and work backwards to workspace root
        for component_id, component_data in components.items():
            signal_file_path = component_data.get('signal_file_path')
            if signal_file_path:
                # signal_file_path is relative to workspace (e.g., "traces/signals/rsi/SPY_rsi_grid_7_20_70.parquet")
                # We need to find the workspace directory that contains this structure
                
                # Look for workspace directories that might contain this file
                workspaces_root = Path(self.workspaces_dir)
                for workspace_dir in workspaces_root.iterdir():
                    if workspace_dir.is_dir():
                        potential_file = workspace_dir / signal_file_path
                        if potential_file.exists():
                            logger.info(f"Found existing signal workspace: {workspace_dir}")
                            return workspace_dir
                            
        return None
    
    def _generate_workspace_name(self, result: Dict[str, Any], topology_name: str, config: Dict[str, Any]) -> str:
        """Generate standardized workspace name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get symbols and experiment info
        symbols = config.get('symbols', ['UNKNOWN'])
        if isinstance(symbols, list):
            symbols_str = '_'.join(symbols)
        else:
            symbols_str = str(symbols)
            
        experiment_id = config.get('metadata', {}).get('experiment_id', topology_name)
        
        return f"{timestamp}_{experiment_id}_{symbols_str}"
    
    def _insert_run_record(self, workspace: AnalyticsWorkspace, 
                          result: Dict[str, Any], topology_name: str, config: Dict[str, Any]) -> str:
        """Insert run record and return run_id"""
        
        # Generate run ID
        run_id = result.get('execution_id', str(uuid.uuid4()))
        
        # Extract run information
        run_data = {
            'run_id': run_id,
            'created_at': datetime.fromisoformat(result.get('start_time', datetime.now().isoformat())),
            'workflow_type': topology_name,
            
            # Data characteristics
            'symbols': config.get('symbols', []),
            'timeframes': config.get('timeframes', []),
            'start_date': config.get('start_date'),
            'end_date': config.get('end_date'),
            'total_bars': config.get('max_bars', config.get('data', {}).get('max_bars')),
            
            # Configuration
            'config_file': str(config.get('name', 'unknown')),
            'config_hash': self._hash_config(config),
            
            # Execution details
            'total_strategies': 0,  # Will be updated later
            'total_classifiers': 0,  # Will be updated later
            'total_combinations': 0,  # Will be updated later
            
            # Status and performance
            'status': 'completed' if result.get('success') else 'failed',
            'duration_seconds': result.get('duration_seconds'),
            'peak_memory_mb': None,  # Could extract from metrics if available
            
            # Storage details
            'workspace_path': str(workspace.workspace_path),
            'total_size_mb': None,  # Could calculate later
            'compression_ratio': None  # Could calculate later
        }
        
        # Insert run record
        columns = ', '.join(run_data.keys())
        placeholders = ', '.join(['?' for _ in run_data])
        
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO runs ({columns})
            VALUES ({placeholders})
        """, list(run_data.values()))
        
        return run_id
    
    def _extract_strategies_data(self, result: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract strategy data from execution result"""
        strategies_data = []
        
        # First, try to get expanded strategies from topology metadata
        topology_metadata = result.get('topology_metadata', {})
        expanded_strategies = topology_metadata.get('expanded_strategies', [])
        
        if expanded_strategies:
            logger.info(f"Found {len(expanded_strategies)} expanded strategies in topology metadata")
            strategies_data = self._create_strategies_from_expanded_config(expanded_strategies)
        else:
            # Fallback: Look for strategy results in outputs or metrics
            metrics = result.get('metrics', {})
            outputs = result.get('outputs', {})
            
            # Try to extract from container metrics
            for container_name, container_metrics in metrics.items():
                if 'strategy' in container_name.lower():
                    # Extract strategy performance metrics
                    strategy_data = self._extract_strategy_metrics(container_name, container_metrics, config)
                    if strategy_data:
                        strategies_data.append(strategy_data)
            
            # If no strategies found in metrics, create from original config
            if not strategies_data:
                strategies_data = self._create_strategies_from_config(config)
        
        return strategies_data
    
    def _extract_classifiers_data(self, result: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract classifier data from execution result"""
        classifiers_data = []
        
        # First, try to get expanded classifiers from topology metadata
        topology_metadata = result.get('topology_metadata', {})
        expanded_classifiers = topology_metadata.get('expanded_classifiers', [])
        
        if expanded_classifiers:
            logger.info(f"Found {len(expanded_classifiers)} expanded classifiers in topology metadata")
            classifiers_data = self._create_classifiers_from_expanded_config(expanded_classifiers)
        else:
            # Fallback: Look for classifier results in outputs or metrics
            metrics = result.get('metrics', {})
            
            # Try to extract from container metrics
            for container_name, container_metrics in metrics.items():
                if 'classifier' in container_name.lower():
                    # Extract classifier metrics
                    classifier_data = self._extract_classifier_metrics(container_name, container_metrics, config)
                    if classifier_data:
                        classifiers_data.append(classifier_data)
            
            # If no classifiers found in metrics, create from original config
            if not classifiers_data:
                classifiers_data = self._create_classifiers_from_config(config)
        
        return classifiers_data
    
    def _extract_strategy_metrics(self, container_name: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract strategy catalog info from container (no performance metrics)"""
        try:
            strategy_id = self._generate_strategy_id(container_name, metrics)
            
            return {
                'strategy_id': strategy_id,
                'strategy_type': metrics.get('strategy_type', 'unknown'),
                'strategy_name': container_name,
                'parameters': json.dumps(metrics.get('parameters', {})),
                'config_hash': self._hash_config(metrics.get('parameters', {})),
                
                # Pure lazy - NO performance metrics stored
                'created_at': datetime.now()
            }
        except Exception as e:
            logger.warning(f"Failed to extract strategy catalog info from {container_name}: {e}")
            return None
    
    def _extract_classifier_metrics(self, container_name: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract classifier catalog info from container (no performance metrics)"""
        try:
            classifier_id = self._generate_classifier_id(container_name, metrics)
            
            return {
                'classifier_id': classifier_id,
                'classifier_type': metrics.get('classifier_type', 'unknown'),
                'classifier_name': container_name,
                'parameters': json.dumps(metrics.get('parameters', {})),
                'config_hash': self._hash_config(metrics.get('parameters', {})),
                
                # Pure lazy - NO classifier statistics stored
                'created_at': datetime.now()
            }
        except Exception as e:
            logger.warning(f"Failed to extract classifier catalog info from {container_name}: {e}")
            return None
    
    def _create_strategies_from_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create strategy records from configuration (when no metrics available)"""
        strategies_data = []
        
        strategies_config = config.get('strategies', [])
        for strategy_config in strategies_config:
            # Handle parameter expansions
            param_combinations = self._expand_parameters(strategy_config.get('params', {}))
            
            for i, params in enumerate(param_combinations):
                strategy_id = f"{strategy_config.get('type', 'unknown')}_{strategy_config.get('name', 'unnamed')}_{i:03d}"
                
                strategies_data.append({
                    'strategy_id': strategy_id,
                    'strategy_type': strategy_config.get('type', 'unknown'),
                    'strategy_name': strategy_config.get('name', 'unnamed'),
                    'parameters': json.dumps(params),
                    'config_hash': self._hash_config(params),
                    
                    # Pure lazy - NO performance metrics stored
                    'created_at': datetime.now()
                })
        
        return strategies_data
    
    def _create_classifiers_from_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create classifier records from configuration (when no metrics available)"""
        classifiers_data = []
        
        classifiers_config = config.get('classifiers', [])
        for classifier_config in classifiers_config:
            # Handle parameter expansions
            param_combinations = self._expand_parameters(classifier_config.get('params', {}))
            
            for i, params in enumerate(param_combinations):
                classifier_id = f"{classifier_config.get('type', 'unknown')}_{classifier_config.get('name', 'unnamed')}_{i:03d}"
                
                classifiers_data.append({
                    'classifier_id': classifier_id,
                    'classifier_type': classifier_config.get('type', 'unknown'),
                    'classifier_name': classifier_config.get('name', 'unnamed'),
                    'parameters': json.dumps(params),
                    'config_hash': self._hash_config(params),
                    
                    'created_at': datetime.now()
                })
        
        return classifiers_data
    
    def _create_strategies_from_expanded_config(self, expanded_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create strategy records from already-expanded strategy configurations"""
        strategies_data = []
        
        for strategy_config in expanded_strategies:
            strategy_id = f"{strategy_config.get('type', 'unknown')}_{strategy_config.get('name', 'unnamed')}"
            
            strategies_data.append({
                'strategy_id': strategy_id,
                'strategy_type': strategy_config.get('type', 'unknown'),
                'strategy_name': strategy_config.get('name', 'unnamed'),
                'parameters': json.dumps(strategy_config.get('params', {})),
                'config_hash': self._hash_config(strategy_config.get('params', {})),
                
                # Pure lazy - NO performance metrics stored
                'created_at': datetime.now()
            })
        
        return strategies_data
    
    def _create_classifiers_from_expanded_config(self, expanded_classifiers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create classifier records from already-expanded classifier configurations"""
        classifiers_data = []
        
        for classifier_config in expanded_classifiers:
            classifier_id = f"{classifier_config.get('type', 'unknown')}_{classifier_config.get('name', 'unnamed')}"
            
            classifiers_data.append({
                'classifier_id': classifier_id,
                'classifier_type': classifier_config.get('type', 'unknown'),
                'classifier_name': classifier_config.get('name', 'unnamed'),
                'parameters': json.dumps(classifier_config.get('params', {})),
                'config_hash': self._hash_config(classifier_config.get('params', {})),
                
                'created_at': datetime.now()
            })
        
        return classifiers_data
    
    def _expand_parameters(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand parameter grid into individual combinations"""
        import itertools
        
        # Separate single values from lists
        single_params = {}
        list_params = {}
        
        for key, value in params.items():
            if isinstance(value, list):
                list_params[key] = value
            else:
                single_params[key] = value
        
        # Generate all combinations of list parameters
        if not list_params:
            return [single_params]
        
        keys = list(list_params.keys())
        values = list(list_params.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combined_params = single_params.copy()
            for key, value in zip(keys, combo):
                combined_params[key] = value
            combinations.append(combined_params)
        
        return combinations
    
    def _generate_strategy_id(self, container_name: str, metrics: Dict[str, Any]) -> str:
        """Generate strategy ID from container name and metrics"""
        strategy_type = metrics.get('strategy_type', 'unknown')
        params_hash = self._hash_config(metrics.get('parameters', {}))[:8]
        return f"{strategy_type}_{container_name}_{params_hash}"
    
    def _generate_classifier_id(self, container_name: str, metrics: Dict[str, Any]) -> str:
        """Generate classifier ID from container name and metrics"""
        classifier_type = metrics.get('classifier_type', 'unknown')
        params_hash = self._hash_config(metrics.get('parameters', {}))[:8]
        return f"{classifier_type}_{container_name}_{params_hash}"
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _insert_strategy_record(self, workspace: AnalyticsWorkspace, strategy_data: Dict[str, Any]) -> None:
        """Insert strategy record into database"""
        columns = ', '.join(strategy_data.keys())
        placeholders = ', '.join(['?' for _ in strategy_data])
        
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO strategies ({columns})
            VALUES ({placeholders})
        """, list(strategy_data.values()))
    
    def _insert_classifier_record(self, workspace: AnalyticsWorkspace, classifier_data: Dict[str, Any]) -> None:
        """Insert classifier record into database"""
        columns = ', '.join(classifier_data.keys())
        placeholders = ', '.join(['?' for _ in classifier_data])
        
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO classifiers ({columns})
            VALUES ({placeholders})
        """, list(classifier_data.values()))


# Global integrator instance
_analytics_integrator = None

def get_analytics_integrator() -> AnalyticsIntegrator:
    """Get or create global analytics integrator"""
    global _analytics_integrator
    if _analytics_integrator is None:
        _analytics_integrator = AnalyticsIntegrator()
    return _analytics_integrator

def integrate_with_topology_result(result: Dict[str, Any], 
                                  topology_name: str, 
                                  config: Dict[str, Any]) -> Optional[Path]:
    """Convenience function to integrate analytics with topology result"""
    integrator = get_analytics_integrator()
    return integrator.create_workspace_for_run(result, topology_name, config)