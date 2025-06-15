#!/usr/bin/env python3
"""
Analyze the SQL database from the grid search run
"""
import duckdb
import pandas as pd
from pathlib import Path

def analyze_sql_database(db_path: str):
    """Analyze the SQL analytics database."""
    
    print(f"ANALYZING SQL DATABASE: {db_path}")
    print("="*80)
    
    # Check if another db file is available
    workspace_path = Path(db_path).parent
    workspace_db = workspace_path / "analytics.duckdb"
    
    if workspace_db.exists():
        db_file = str(workspace_db)
        print(f"Using workspace database: {db_file}")
    else:
        db_file = db_path
    
    try:
        conn = duckdb.connect(db_file, read_only=True)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        # Try copying the file first
        import shutil
        temp_db = "temp_analytics.duckdb"
        shutil.copy(db_file, temp_db)
        conn = duckdb.connect(temp_db, read_only=True)
        print(f"Using temporary copy: {temp_db}")
    
    # Check tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"\nTABLES IN DATABASE:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Analyze strategies table
    print(f"\nSTRATEGIES TABLE ANALYSIS:")
    strategy_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    print(f"  Total strategies: {strategy_count}")
    
    # Strategy types
    strategy_types = conn.execute("""
        SELECT strategy_type, COUNT(*) as count
        FROM strategies
        GROUP BY strategy_type
        ORDER BY count DESC
    """).fetchall()
    
    print(f"\n  Strategy types ({len(strategy_types)}):")
    for stype, count in strategy_types:
        print(f"    {stype:<35} → {count:3d} configurations")
    
    # Check for missing strategies
    expected_strategies = [
        'sma_crossover', 'ema_crossover', 'macd_crossover', 'stochastic_crossover',
        'rsi_threshold', 'rsi_bands', 'cci_threshold', 'cci_bands',
        'bollinger_breakout', 'keltner_breakout', 'donchian_breakout',
        'obv_trend', 'mfi_bands', 'vwap_deviation', 'chaikin_money_flow',
        'adx_trend_strength', 'parabolic_sar', 'aroon_crossover', 'supertrend',
        'pivot_points', 'fibonacci_retracement', 'support_resistance_breakout',
        'stochastic_rsi', 'williams_r', 'roc_threshold', 'ultimate_oscillator'
    ]
    
    loaded_types = [s[0].replace('_grid', '') for s in strategy_types]
    missing = [s for s in expected_strategies if s not in loaded_types]
    
    if missing:
        print(f"\n  MISSING STRATEGIES ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")
    
    # Analyze classifiers table
    print(f"\nCLASSIFIERS TABLE ANALYSIS:")
    classifier_count = conn.execute("SELECT COUNT(*) FROM classifiers").fetchone()[0]
    print(f"  Total classifiers: {classifier_count}")
    
    # Classifier types
    classifier_types = conn.execute("""
        SELECT classifier_type, COUNT(*) as count
        FROM classifiers
        GROUP BY classifier_type
        ORDER BY count DESC
    """).fetchall()
    
    print(f"\n  Classifier types ({len(classifier_types)}):")
    for ctype, count in classifier_types:
        print(f"    {ctype:<35} → {count:3d} configurations")
    
    # Sample some strategy parameters
    print(f"\nSAMPLE STRATEGY PARAMETERS:")
    samples = conn.execute("""
        SELECT strategy_type, strategy_name, parameters
        FROM strategies
        LIMIT 5
    """).fetchall()
    
    for stype, name, params in samples:
        print(f"  {name}: {params}")
    
    # Check metadata
    if 'metadata' in [t[0] for t in tables]:
        print(f"\nMETADATA:")
        metadata = conn.execute("SELECT key, value FROM metadata").fetchall()
        for key, value in metadata:
            print(f"  {key}: {value}")
    
    conn.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    # Analyze the specific run
    db_path = "workspaces/20250614_021319_indicator_grid_v3_SPY/analytics.duckdb"
    analyze_sql_database(db_path)