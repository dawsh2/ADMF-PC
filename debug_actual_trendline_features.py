#!/usr/bin/env python3

"""Debug actual trendline feature keys from a real strategy run."""

import sys
sys.path.append('/Users/daws/ADMF-PC/src')

import sqlite3
from pathlib import Path

def check_actual_trendline_features():
    """Check what trendline features are actually available from strategy runs."""
    
    # Find a recent workspace
    workspaces = list(Path('/Users/daws/ADMF-PC/workspaces').glob('complete_strategy_grid_*'))
    if not workspaces:
        workspaces = list(Path('/Users/daws/ADMF-PC/workspaces').glob('signal_generation_*'))
    if not workspaces:
        print("No workspaces found")
        return
    
    latest_workspace = max(workspaces, key=lambda p: p.stat().st_mtime)
    print(f"Checking workspace: {latest_workspace}")
    
    # Look for features database
    features_db = latest_workspace / 'features.db'
    if not features_db.exists():
        print("No features.db found")
        return
    
    # Check what feature keys are actually stored
    conn = sqlite3.connect(str(features_db))
    cursor = conn.cursor()
    
    # Get all feature names that contain 'trendlines'
    cursor.execute("SELECT DISTINCT feature_name FROM features WHERE feature_name LIKE '%trendlines%' LIMIT 20")
    trendline_features = cursor.fetchall()
    
    print(f"Found {len(trendline_features)} trendline feature keys:")
    for feature, in trendline_features:
        print(f"  {feature}")
    
    if trendline_features:
        # Get some sample values
        sample_feature = trendline_features[0][0]
        cursor.execute("SELECT feature_value FROM features WHERE feature_name = ? AND feature_value IS NOT NULL LIMIT 5", (sample_feature,))
        values = cursor.fetchall()
        print(f"\nSample values for {sample_feature}:")
        for value, in values:
            print(f"  {value}")
    
    conn.close()
    
    # Also check what strategy execution expects
    print("\n" + "="*60)
    print("Checking what strategy execution code expects...")
    
    # Test trendline_breaks strategy expectations
    params_1 = {'pivot_lookback': 20, 'tolerance': 0.002}
    expected_key_1 = f'trendlines_{params_1["pivot_lookback"]}_2_{params_1["tolerance"]}'
    expected_subkeys_1 = [
        f'{expected_key_1}_valid_uptrends',
        f'{expected_key_1}_valid_downtrends', 
        f'{expected_key_1}_nearest_support',
        f'{expected_key_1}_nearest_resistance'
    ]
    
    print(f"\ntrendline_breaks strategy expects:")
    for key in expected_subkeys_1:
        print(f"  {key}")
    
    # Test trendline_bounces strategy expectations  
    params_2 = {'pivot_lookback': 20, 'min_touches': 3, 'tolerance': 0.002}
    expected_key_2 = f'trendlines_{params_2["pivot_lookback"]}_{params_2["min_touches"]}_{params_2["tolerance"]}'
    expected_subkeys_2 = [
        f'{expected_key_2}_nearest_support',
        f'{expected_key_2}_nearest_resistance',
        f'{expected_key_2}_valid_uptrends',
        f'{expected_key_2}_valid_downtrends'
    ]
    
    print(f"\ntrendline_bounces strategy expects:")
    for key in expected_subkeys_2:
        print(f"  {key}")
    
    # Check if any of these match what we found
    if trendline_features:
        available_keys = set(f[0] for f in trendline_features)
        
        print(f"\nMatching analysis:")
        print(f"trendline_breaks matches: {sum(1 for k in expected_subkeys_1 if k in available_keys)}/{len(expected_subkeys_1)}")
        print(f"trendline_bounces matches: {sum(1 for k in expected_subkeys_2 if k in available_keys)}/{len(expected_subkeys_2)}")
        
        # Show what's missing
        missing_1 = [k for k in expected_subkeys_1 if k not in available_keys]
        missing_2 = [k for k in expected_subkeys_2 if k not in available_keys]
        
        if missing_1:
            print(f"\ntrendline_breaks missing keys:")
            for key in missing_1:
                print(f"  {key}")
                
        if missing_2:
            print(f"\ntrendline_bounces missing keys:")
            for key in missing_2:
                print(f"  {key}")

if __name__ == "__main__":
    check_actual_trendline_features()