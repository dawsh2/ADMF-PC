#!/usr/bin/env python3
"""
Batch analysis of strategies by regime using reusable SQL queries
Processes strategies in batches and calculates proper daily Sharpe ratios
"""

import subprocess
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import time
import os

# Configuration
ANALYTICS_DB = "analytics.duckdb"
QUERIES_DIR = Path(__file__).parent
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Analysis parameters
START_DATE = "2024-03-26 00:00:00"
END_DATE = "2025-01-17 20:00:00"
CLASSIFIER_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet"
MARKET_DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"

def run_duckdb_query(query_file, variables=None, db_path=ANALYTICS_DB):
    """Execute a DuckDB query file with variables"""
    # Build command
    cmd = ['duckdb', db_path]
    
    # Add variables
    if variables:
        for key, value in variables.items():
            cmd.extend(['-variable', f"{key}={value}"])
    
    # Add query file
    cmd.extend(['-c', f".read {query_file}"])
    
    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error executing {query_file}: {result.stderr}")
        return None
    
    return result.stdout

def analyze_strategy(strategy_info, regime_setup_done=False):
    """Analyze a single strategy using SQL queries"""
    
    # Setup regime analysis if not done
    if not regime_setup_done:
        print("Setting up regime analysis tables...")
        variables = {
            'start_date': START_DATE,
            'end_date': END_DATE,
            'classifier_path': CLASSIFIER_PATH,
            'market_data_path': MARKET_DATA_PATH
        }
        run_duckdb_query(QUERIES_DIR / "01_setup_regime_analysis.sql", variables)
    
    # Analyze the strategy with daily Sharpe
    print(f"  Analyzing {strategy_info['strategy_name']}...")
    variables = {
        'strategy_path': strategy_info['signal_file_path'],
        'strategy_id': strategy_info['strategy_id'],
        'strategy_name': strategy_info['strategy_name'],
        'start_date': START_DATE,
        'end_date': END_DATE,
        'risk_free_rate': 0.0
    }
    
    # Run the daily Sharpe analysis
    output = run_duckdb_query(QUERIES_DIR / "04_calculate_daily_sharpe.sql", variables)
    
    if output:
        # Parse output and return results
        # This is simplified - real implementation would parse DuckDB output
        return {
            'strategy_id': strategy_info['strategy_id'],
            'strategy_name': strategy_info['strategy_name'],
            'strategy_type': strategy_info['strategy_type'],
            'analysis_complete': True
        }
    
    return None

def get_strategies_to_analyze(strategy_type=None, limit=None):
    """Get list of strategies from database"""
    query = f"""
    SELECT 
        strategy_id,
        strategy_name,
        strategy_type,
        signal_file_path
    FROM analytics.strategies
    WHERE signal_file_path IS NOT NULL
    """
    
    if strategy_type:
        query += f" AND strategy_type = '{strategy_type}'"
    
    if limit:
        query += f" LIMIT {limit}"
    
    # Execute query
    cmd = ['duckdb', ANALYTICS_DB, '-csv', '-c', query]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error getting strategies: {result.stderr}")
        return []
    
    # Parse CSV output
    from io import StringIO
    df = pd.read_csv(StringIO(result.stdout))
    return df.to_dict('records')

def main():
    """Main analysis function"""
    print(f"Starting comprehensive strategy analysis")
    print(f"Period: {START_DATE} to {END_DATE}")
    
    # Get strategies to analyze
    strategies = get_strategies_to_analyze(limit=100)  # Start with first 100
    print(f"Found {len(strategies)} strategies to analyze")
    
    # Group by type for efficient processing
    from collections import defaultdict
    strategies_by_type = defaultdict(list)
    for s in strategies:
        strategies_by_type[s['strategy_type']].append(s)
    
    # Process each type
    all_results = []
    regime_setup_done = False
    
    for strategy_type, type_strategies in strategies_by_type.items():
        print(f"\nAnalyzing {strategy_type} strategies ({len(type_strategies)} total)...")
        
        for i, strategy in enumerate(type_strategies[:10]):  # Limit to 10 per type for testing
            try:
                result = analyze_strategy(strategy, regime_setup_done)
                regime_setup_done = True  # Only setup once
                
                if result:
                    all_results.append(result)
                    
                # Progress update
                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{len(type_strategies)}")
                    
            except Exception as e:
                print(f"  Error analyzing {strategy['strategy_name']}: {str(e)}")
            
            # Brief pause to avoid overwhelming system
            time.sleep(0.1)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = RESULTS_DIR / f"strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Run summary analysis
        print("\nRunning summary analysis...")
        run_duckdb_query(
            QUERIES_DIR / "05_regime_performance_summary.sql",
            {'results_csv': str(output_file)}
        )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()