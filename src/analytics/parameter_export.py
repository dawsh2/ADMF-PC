"""
Parameter export utilities for walk-forward validation.

Provides functionality to export selected parameters from SQL analysis
for use in subsequent WFV phases.
"""

import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def export_selected_parameters(workspace_path: str, 
                              selection_query: str, 
                              output_file: str) -> None:
    """
    Export parameters selected via SQL analysis.
    
    Args:
        workspace_path: Path to workspace directory containing analytics.db
        selection_query: SQL query to select optimal parameters
        output_file: Output JSON file path for selected parameters
        
    Raises:
        FileNotFoundError: If workspace or database doesn't exist
        sqlite3.Error: If SQL query fails
    """
    workspace_dir = Path(workspace_path)
    if not workspace_dir.exists():
        raise FileNotFoundError(f"Workspace directory not found: {workspace_path}")
    
    db_path = workspace_dir / 'analytics.db'
    if not db_path.exists():
        raise FileNotFoundError(f"Analytics database not found: {db_path}")
    
    try:
        # Connect to database and execute query
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            logger.info(f"Executing selection query: {selection_query}")
            cursor.execute(selection_query)
            results = cursor.fetchall()
            
            if not results:
                logger.warning("Selection query returned no results")
                return
            
            # Convert results to parameter configurations
            selected_params = _convert_results_to_parameters(results)
            
            # Export to JSON file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(selected_params, f, indent=2)
            
            logger.info(f"Exported {len(selected_params.get('strategies', []))} strategies "
                       f"and {len(selected_params.get('classifiers', []))} classifiers "
                       f"to {output_file}")
                       
    except sqlite3.Error as e:
        logger.error(f"Database error during parameter export: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to export parameters: {e}")
        raise


def _convert_results_to_parameters(results: List[sqlite3.Row]) -> Dict[str, Any]:
    """
    Convert SQL query results to parameter configuration format.
    
    Args:
        results: List of database rows from selection query
        
    Returns:
        Parameter configuration dictionary
    """
    strategies = []
    classifiers = []
    
    for row in results:
        # Determine if this is a strategy or classifier
        component_type = _determine_component_type(row)
        
        if component_type == 'strategy':
            strategy_config = _extract_strategy_config(row)
            if strategy_config:
                strategies.append(strategy_config)
        elif component_type == 'classifier':
            classifier_config = _extract_classifier_config(row)
            if classifier_config:
                classifiers.append(classifier_config)
    
    return {
        'strategies': strategies,
        'classifiers': classifiers
    }


def _determine_component_type(row: sqlite3.Row) -> str:
    """
    Determine if a database row represents a strategy or classifier.
    
    Args:
        row: Database row
        
    Returns:
        Component type ('strategy' or 'classifier')
    """
    # Convert row to dict for easier access
    row_dict = dict(row)
    
    # Check for strategy-specific columns
    if 'strategy_name' in row_dict or 'strategy_type' in row_dict:
        return 'strategy'
    elif 'classifier_name' in row_dict or 'classifier_type' in row_dict:
        return 'classifier'
    else:
        # Default to strategy for backwards compatibility
        return 'strategy'


def _extract_strategy_config(row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    """
    Extract strategy configuration from database row.
    
    Args:
        row: Database row containing strategy data
        
    Returns:
        Strategy configuration dictionary
    """
    try:
        # Convert row to dict for easier access
        row_dict = dict(row)
        
        # Extract basic strategy info
        strategy_name = row_dict.get('strategy_name') or row_dict.get('name')
        strategy_type = row_dict.get('strategy_type') or row_dict.get('type')
        
        if not strategy_name or not strategy_type:
            logger.warning(f"Missing strategy name/type in row: {row_dict}")
            return None
        
        # Extract parameters (stored as JSON string or individual columns)
        params = {}
        params_json = row_dict.get('strategy_params') or row_dict.get('params')
        
        if params_json:
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in params: {params_json}")
        
        # Check for individual parameter columns
        for key in row_dict.keys():
            if key.startswith('param_') or key in ['lookback', 'threshold', 'period', 'fast_period', 'slow_period']:
                param_name = key.replace('param_', '')
                params[param_name] = row_dict[key]
        
        return {
            'name': strategy_name,
            'type': strategy_type,
            'params': params
        }
        
    except Exception as e:
        logger.error(f"Failed to extract strategy config from row: {e}")
        return None


def _extract_classifier_config(row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    """
    Extract classifier configuration from database row.
    
    Args:
        row: Database row containing classifier data
        
    Returns:
        Classifier configuration dictionary
    """
    try:
        # Convert row to dict for easier access
        row_dict = dict(row)
        
        # Extract basic classifier info
        classifier_name = row_dict.get('classifier_name') or row_dict.get('name')
        classifier_type = row_dict.get('classifier_type') or row_dict.get('type')
        
        if not classifier_name or not classifier_type:
            logger.warning(f"Missing classifier name/type in row: {row_dict}")
            return None
        
        # Extract parameters
        params = {}
        params_json = row_dict.get('classifier_params') or row_dict.get('params')
        
        if params_json:
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in params: {params_json}")
        
        # Check for individual parameter columns
        for key in row_dict.keys():
            if key.startswith('param_') or key in ['fast_ma', 'slow_ma', 'threshold', 'period']:
                param_name = key.replace('param_', '')
                params[param_name] = row_dict[key]
        
        return {
            'name': classifier_name,
            'type': classifier_type,
            'params': params
        }
        
    except Exception as e:
        logger.error(f"Failed to extract classifier config from row: {e}")
        return None


def export_generation_parameters(workspace_path: str, 
                                generation_num: int,
                                output_file: str) -> None:
    """
    Export specific GA generation parameters.
    
    Args:
        workspace_path: Path to workspace directory
        generation_num: Generation number to export
        output_file: Output JSON file path
        
    Raises:
        FileNotFoundError: If workspace or generation file doesn't exist
    """
    workspace_dir = Path(workspace_path)
    if not workspace_dir.exists():
        raise FileNotFoundError(f"Workspace directory not found: {workspace_path}")
    
    generations_dir = workspace_dir / 'generations'
    if not generations_dir.exists():
        raise FileNotFoundError(f"Generations directory not found: {generations_dir}")
    
    gen_file = generations_dir / f"gen_{generation_num:02d}.json"
    if not gen_file.exists():
        raise FileNotFoundError(f"Generation file not found: {gen_file}")
    
    try:
        # Load generation parameters
        with open(gen_file, 'r') as f:
            generation_params = json.load(f)
        
        # Export to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(generation_params, f, indent=2)
        
        logger.info(f"Exported generation {generation_num} parameters to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to export generation parameters: {e}")
        raise


def analyze_generation_robustness(workspace_path: str, 
                                  last_n_generations: int = 10) -> Dict[str, Any]:
    """
    Analyze last N generations for optimal stopping selection.
    
    Args:
        workspace_path: Path to workspace directory
        last_n_generations: Number of recent generations to analyze
        
    Returns:
        Analysis results with robustness metrics
        
    Raises:
        FileNotFoundError: If workspace doesn't exist
    """
    workspace_dir = Path(workspace_path)
    if not workspace_dir.exists():
        raise FileNotFoundError(f"Workspace directory not found: {workspace_path}")
    
    db_path = workspace_dir / 'analytics.db'
    if not db_path.exists():
        raise FileNotFoundError(f"Analytics database not found: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Query generation performance data
            query = """
            SELECT generation, avg_validation_sharpe, avg_validation_return, 
                   validation_consistency, train_validation_gap
            FROM generation_analysis 
            ORDER BY generation DESC 
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[last_n_generations])
            
            if df.empty:
                logger.warning("No generation analysis data found")
                return {}
            
            # Calculate robustness metrics
            analysis = {
                'generations_analyzed': len(df),
                'best_validation_sharpe': {
                    'generation': int(df.loc[df['avg_validation_sharpe'].idxmax(), 'generation']),
                    'sharpe': float(df['avg_validation_sharpe'].max())
                },
                'most_consistent': {
                    'generation': int(df.loc[df['validation_consistency'].idxmax(), 'generation']),
                    'consistency': float(df['validation_consistency'].max())
                },
                'smallest_overfitting': {
                    'generation': int(df.loc[df['train_validation_gap'].idxmin(), 'generation']),
                    'gap': float(df['train_validation_gap'].min())
                },
                'recommended_generation': None
            }
            
            # Simple recommendation: balance validation performance and consistency
            df['robustness_score'] = (df['avg_validation_sharpe'] * 0.4 + 
                                    df['validation_consistency'] * 0.4 - 
                                    df['train_validation_gap'] * 0.2)
            
            best_idx = df['robustness_score'].idxmax()
            analysis['recommended_generation'] = {
                'generation': int(df.loc[best_idx, 'generation']),
                'robustness_score': float(df.loc[best_idx, 'robustness_score']),
                'rationale': 'Balanced validation performance, consistency, and low overfitting'
            }
            
            logger.info(f"Generation robustness analysis complete. "
                       f"Recommended generation: {analysis['recommended_generation']['generation']}")
            
            return analysis
            
    except Exception as e:
        logger.error(f"Failed to analyze generation robustness: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export parameters from WFV analysis")
    parser.add_argument('--workspace', required=True, help='Workspace directory path')
    parser.add_argument('--query', help='SQL query for parameter selection')
    parser.add_argument('--generation', type=int, help='Generation number to export')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--analyze-generations', action='store_true', 
                       help='Analyze generation robustness')
    
    args = parser.parse_args()
    
    if args.analyze_generations:
        analysis = analyze_generation_robustness(args.workspace)
        print(json.dumps(analysis, indent=2))
    elif args.generation is not None:
        export_generation_parameters(args.workspace, args.generation, args.output)
    elif args.query:
        export_selected_parameters(args.workspace, args.query, args.output)
    else:
        print("Must specify --query, --generation, or --analyze-generations")