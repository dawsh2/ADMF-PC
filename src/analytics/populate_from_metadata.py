"""
Populate analytics database from metadata.json files.

This module provides functionality to read metadata.json files created by
MultiStrategyTracer and populate the analytics database with properly
structured strategy and classifier information.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib

from .workspace import AnalyticsWorkspace
from .exceptions import AnalyticsError

logger = logging.getLogger(__name__)


def populate_from_metadata(workspace: AnalyticsWorkspace, metadata_path: Union[str, Path]) -> Dict[str, int]:
    """
    Populate analytics database from a metadata.json file.
    
    This function reads the metadata.json file created by MultiStrategyTracer
    and populates the analytics database with strategy and classifier information,
    using the structured fields (strategy_type, parameters) instead of parsing
    strategy_id strings.
    
    Args:
        workspace: AnalyticsWorkspace instance
        metadata_path: Path to metadata.json file
        
    Returns:
        Dict with counts of populated records
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise AnalyticsError(f"Metadata file not found: {metadata_path}")
    
    logger.info(f"Populating analytics from: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create or get run record
    run_id = _create_run_record(workspace, metadata, metadata_path)
    
    # Extract workspace path for relative path resolution
    workspace_base = metadata_path.parent
    
    # Process components
    strategies_count = 0
    classifiers_count = 0
    
    components = metadata.get('components', {})
    for component_id, component_data in components.items():
        component_type = component_data.get('component_type')
        
        if component_type == 'strategy':
            _insert_strategy_from_metadata(workspace, component_id, component_data, run_id, workspace_base)
            strategies_count += 1
        elif component_type == 'classifier':
            _insert_classifier_from_metadata(workspace, component_id, component_data, run_id, workspace_base)
            classifiers_count += 1
        else:
            logger.warning(f"Unknown component type: {component_type} for {component_id}")
    
    # Update run record with counts
    workspace.conn.execute("""
        UPDATE runs SET 
            total_strategies = ?, 
            total_classifiers = ?,
            total_combinations = ?
        WHERE run_id = ?
    """, [strategies_count, classifiers_count, strategies_count * classifiers_count, run_id])
    
    logger.info(f"Populated {strategies_count} strategies and {classifiers_count} classifiers")
    
    return {
        'strategies': strategies_count,
        'classifiers': classifiers_count,
        'run_id': run_id
    }


def _create_run_record(workspace: AnalyticsWorkspace, metadata: Dict[str, Any], metadata_path: Path) -> str:
    """Create run record from metadata."""
    # Generate run ID from workflow_id or path
    run_id = metadata.get('workflow_id', metadata_path.parent.name)
    
    # Parse workspace name for metadata
    workspace_name = metadata_path.parent.name
    parts = workspace_name.split('_')
    
    # Try to extract date from workspace name
    created_at = datetime.now()
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
        # Format: YYYYMMDD_HHMMSS_...
        try:
            date_str = f"{parts[0]}_{parts[1]}"
            created_at = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        except:
            pass
    
    run_data = {
        'run_id': run_id,
        'created_at': created_at,
        'workflow_type': metadata.get('workflow_id', 'unknown'),
        
        # Data characteristics - would need to be extracted from config or data files
        'symbols': [],  # Could extract from component names or signal files
        'timeframes': [],  # Could extract from component names
        'start_date': None,
        'end_date': None,
        'total_bars': metadata.get('total_bars', 0),
        
        # Configuration
        'config_file': workspace_name,
        'config_hash': _hash_config(metadata),
        
        # Execution details
        'total_strategies': 0,  # Will be updated later
        'total_classifiers': 0,  # Will be updated later
        'total_combinations': 0,  # Will be updated later
        
        # Status and performance
        'status': 'completed',
        'duration_seconds': None,
        'peak_memory_mb': None,
        
        # Storage details
        'workspace_path': str(metadata_path.parent),
        'total_size_mb': None,
        'compression_ratio': metadata.get('compression_ratio', 0)
    }
    
    # Insert run record
    columns = ', '.join(run_data.keys())
    placeholders = ', '.join(['?' for _ in run_data])
    
    try:
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO runs ({columns})
            VALUES ({placeholders})
        """, list(run_data.values()))
    except Exception as e:
        logger.error(f"Failed to insert run record: {e}")
        raise
    
    return run_id


def _insert_strategy_from_metadata(workspace: AnalyticsWorkspace, component_id: str, 
                                  component_data: Dict[str, Any], run_id: str, 
                                  workspace_base: Path) -> None:
    """Insert strategy record from metadata component."""
    # Use structured fields from metadata
    strategy_type = component_data.get('strategy_type', 'unknown')
    parameters = component_data.get('parameters', {})
    
    # Build signal file path
    signal_file_path = component_data.get('signal_file_path')
    if signal_file_path:
        # Convert to absolute path if needed
        signal_path = workspace_base / signal_file_path
        if not signal_path.exists():
            logger.warning(f"Signal file not found: {signal_path}")
    
    strategy_record = {
        'strategy_id': component_id,
        'run_id': run_id,
        'strategy_type': strategy_type,
        'strategy_name': component_id,  # Could be enhanced with better naming
        'parameters': json.dumps(parameters),
        'signal_file_path': signal_file_path,
        'config_hash': _hash_config(parameters),
        
        # Performance metrics from metadata
        'total_bars': component_data.get('total_bars'),
        'signal_changes': component_data.get('signal_changes'),
        'signal_frequency': component_data.get('signal_frequency'),
        'compression_ratio': component_data.get('compression_ratio'),
        
        # These would need to be calculated from signal analysis
        'total_return': None,
        'annualized_return': None,
        'volatility': None,
        'sharpe_ratio': None,
        'max_drawdown': None,
        'total_trades': None,
        'win_rate': None,
        
        'created_at': datetime.fromisoformat(component_data.get('created_at', datetime.now().isoformat())),
        'processed_at': datetime.now()
    }
    
    # Insert into database
    columns = ', '.join(strategy_record.keys())
    placeholders = ', '.join(['?' for _ in strategy_record])
    
    try:
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO strategies ({columns})
            VALUES ({placeholders})
        """, list(strategy_record.values()))
    except Exception as e:
        logger.error(f"Failed to insert strategy {component_id}: {e}")
        raise


def _insert_classifier_from_metadata(workspace: AnalyticsWorkspace, component_id: str,
                                    component_data: Dict[str, Any], run_id: str,
                                    workspace_base: Path) -> None:
    """Insert classifier record from metadata component."""
    # Use structured fields from metadata
    classifier_type = component_data.get('strategy_type', 'unknown')  # Note: classifiers use strategy_type field
    parameters = component_data.get('parameters', {})
    
    # Build states file path
    states_file_path = component_data.get('signal_file_path')  # Classifiers also use signal_file_path
    if states_file_path:
        # Convert to absolute path if needed
        states_path = workspace_base / states_file_path
        if not states_path.exists():
            logger.warning(f"States file not found: {states_path}")
    
    classifier_record = {
        'classifier_id': component_id,
        'run_id': run_id,
        'classifier_type': classifier_type,
        'classifier_name': component_id,
        'parameters': json.dumps(parameters),
        'states_file_path': states_file_path,
        'config_hash': _hash_config(parameters),
        
        # Metadata from component
        'total_bars': component_data.get('total_bars'),
        'state_changes': component_data.get('signal_changes'),  # Signal changes = state changes
        'compression_ratio': component_data.get('compression_ratio'),
        
        # These would need to be calculated from states analysis
        'total_states': None,
        'regime_counts': None,
        'avg_regime_duration': None,
        'transition_matrix': None,
        'stability_score': None,
        
        'created_at': datetime.fromisoformat(component_data.get('created_at', datetime.now().isoformat())),
        'processed_at': datetime.now()
    }
    
    # Insert into database
    columns = ', '.join(classifier_record.keys())
    placeholders = ', '.join(['?' for _ in classifier_record])
    
    try:
        workspace.conn.execute(f"""
            INSERT OR REPLACE INTO classifiers ({columns})
            VALUES ({placeholders})
        """, list(classifier_record.values()))
    except Exception as e:
        logger.error(f"Failed to insert classifier {component_id}: {e}")
        raise


def _hash_config(config: Union[Dict[str, Any], Any]) -> str:
    """Generate hash for configuration."""
    if isinstance(config, dict):
        config_str = json.dumps(config, sort_keys=True)
    else:
        config_str = str(config)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def populate_workspace_from_traces(workspace_path: Union[str, Path]) -> None:
    """
    Convenience function to populate a workspace from its metadata.json file.
    
    Args:
        workspace_path: Path to workspace directory containing metadata.json
    """
    workspace_path = Path(workspace_path)
    metadata_path = workspace_path / 'metadata.json'
    
    if not metadata_path.exists():
        raise AnalyticsError(f"No metadata.json found in {workspace_path}")
    
    # Create or open analytics workspace
    workspace = AnalyticsWorkspace(workspace_path)
    
    try:
        result = populate_from_metadata(workspace, metadata_path)
        logger.info(f"Successfully populated workspace with {result['strategies']} strategies "
                   f"and {result['classifiers']} classifiers")
    finally:
        workspace.close()