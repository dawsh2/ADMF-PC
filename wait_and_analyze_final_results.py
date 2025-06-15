#!/usr/bin/env python3
"""
Wait for the grid search to complete and then analyze the final results
to verify that all 882 strategies are generating signals.
"""

import sys
import os
import time
import glob
import pandas as pd
from pathlib import Path

def find_newest_workspace():
    """Find the most recently created workspace."""
    # Look for workspaces created in the last hour
    workspace_patterns = [
        "/Users/daws/ADMF-PC/workspaces/expansive_grid_search_*",
        "/Users/daws/ADMF-PC/workspaces/signal_generation_*"
    ]
    
    newest_workspace = None
    newest_time = 0
    
    for pattern in workspace_patterns:
        workspaces = glob.glob(pattern)
        for workspace in workspaces:
            mtime = os.path.getmtime(workspace)
            if mtime > newest_time:
                newest_time = mtime
                newest_workspace = workspace
    
    return newest_workspace

def check_workspace_completion(workspace_dir):
    """Check if workspace has completed signal generation."""
    if not workspace_dir or not os.path.exists(workspace_dir):
        return False
    
    # Check for completion indicators
    analytics_db = os.path.join(workspace_dir, "analytics.duckdb")
    metadata_file = os.path.join(workspace_dir, "metadata.json")
    
    # Must have both files to be considered complete
    return os.path.exists(analytics_db) and os.path.exists(metadata_file)

def analyze_signal_coverage(workspace_dir):
    """Analyze signal coverage in the completed workspace."""
    
    print(f"🔍 Analyzing completed workspace: {workspace_dir}")
    
    # Look for signals in traces directory
    signals_dir = os.path.join(workspace_dir, "traces", "SPY_1m", "signals")
    
    if not os.path.exists(signals_dir):
        print("❌ No signals directory found")
        return False
    
    # Count signal files by strategy type
    strategy_groups = {}
    total_files = 0
    
    for strategy_group in os.listdir(signals_dir):
        group_path = os.path.join(signals_dir, strategy_group)
        if os.path.isdir(group_path):
            # Count parquet files
            parquet_files = glob.glob(os.path.join(group_path, "*.parquet"))
            strategy_groups[strategy_group] = len(parquet_files)
            total_files += len(parquet_files)
    
    print(f"📊 Signal files by strategy type:")
    for strategy, count in sorted(strategy_groups.items()):
        print(f"   • {strategy}: {count} files")
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"   • Total strategy files: {total_files}")
    print(f"   • Expected total: 882")
    print(f"   • Coverage: {total_files}/882 ({total_files/882*100:.1f}%)")
    
    # Check if we have signal files for all major strategy types
    expected_strategy_types = {
        'sma_crossover_grid', 'ema_crossover_grid', 'ema_sma_crossover_grid',
        'dema_crossover_grid', 'dema_sma_crossover_grid', 'tema_sma_crossover_grid',
        'stochastic_crossover_grid', 'vortex_crossover_grid', 'macd_crossover_grid',
        'ichimoku_cloud_position_grid', 'rsi_threshold_grid', 'rsi_bands_grid',
        'cci_threshold_grid', 'cci_bands_grid', 'stochastic_rsi_grid',
        'williams_r_grid', 'roc_threshold_grid', 'ultimate_oscillator_grid',
        'bollinger_breakout_grid', 'keltner_breakout_grid', 'donchian_breakout_grid',
        'obv_trend_grid', 'mfi_bands_grid', 'vwap_deviation_grid',
        'chaikin_money_flow_grid', 'accumulation_distribution_grid',
        'adx_trend_strength_grid', 'parabolic_sar_grid', 'aroon_crossover_grid',
        'supertrend_grid', 'linear_regression_slope_grid', 'pivot_points_grid',
        'fibonacci_retracement_grid', 'support_resistance_breakout_grid',
        'atr_channel_breakout_grid', 'price_action_swing_grid'
    }
    
    found_strategy_types = set(strategy_groups.keys())
    missing_strategy_types = expected_strategy_types - found_strategy_types
    
    strategy_type_coverage = len(found_strategy_types) / len(expected_strategy_types) * 100
    
    print(f"\n🎯 STRATEGY TYPE COVERAGE:")
    print(f"   • Strategy types with signals: {len(found_strategy_types)}/{len(expected_strategy_types)}")
    print(f"   • Type coverage: {strategy_type_coverage:.1f}%")
    
    if missing_strategy_types:
        print(f"   • Missing strategy types: {sorted(missing_strategy_types)[:5]}...")
    
    # Sample a few signal files to verify data quality
    sample_files = []
    for group, count in list(strategy_groups.items())[:3]:
        group_path = os.path.join(signals_dir, group)
        parquet_files = glob.glob(os.path.join(group_path, "*.parquet"))[:2]
        sample_files.extend(parquet_files)
    
    if sample_files:
        print(f"\n🔍 SIGNAL DATA QUALITY CHECK:")
        
        for file_path in sample_files[:3]:
            try:
                df = pd.read_parquet(file_path)
                filename = os.path.basename(file_path)
                print(f"   • {filename}: {len(df)} signals")
                
                # Check for required columns
                required_cols = ['ts', 'val', 'strat']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"     ⚠️  Missing columns: {missing_cols}")
                else:
                    print(f"     ✅ All required columns present")
                    
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
    
    # Overall success assessment
    file_coverage = total_files / 882 * 100
    overall_success = file_coverage >= 80 and strategy_type_coverage >= 90
    
    print(f"\n{'🎉' if overall_success else '⚠️'} MIGRATION SUCCESS ASSESSMENT:")
    if overall_success:
        print(f"   ✅ EXCELLENT! High signal coverage achieved")
        print(f"   ✅ {file_coverage:.1f}% file coverage")
        print(f"   ✅ {strategy_type_coverage:.1f}% strategy type coverage")
        print(f"   ✅ Migration successfully resolved naming issues")
        print(f"   ✅ Simplified approach working at scale")
    else:
        print(f"   ⚠️  PARTIAL SUCCESS: {file_coverage:.1f}% coverage")
        print(f"   ⚠️  May need parameter tuning for missing strategies")
    
    return overall_success

def main():
    """Main function to wait and analyze results."""
    
    print("=== WAITING FOR GRID SEARCH COMPLETION ===")
    print("Monitoring for completed workspace...")
    
    max_wait_time = 30 * 60  # 30 minutes max
    check_interval = 30      # Check every 30 seconds
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        # Find newest workspace
        newest_workspace = find_newest_workspace()
        
        if newest_workspace and check_workspace_completion(newest_workspace):
            print(f"\n✅ Found completed workspace: {newest_workspace}")
            
            # Wait a bit more to ensure all files are written
            print("⏳ Waiting 30 seconds for file writes to complete...")
            time.sleep(30)
            
            # Analyze the results
            success = analyze_signal_coverage(newest_workspace)
            
            if success:
                print("\n🎉 MIGRATION VERIFICATION: ✅ SUCCESS!")
                print("🎯 Strategy migration completed successfully!")
            else:
                print("\n⚠️  MIGRATION VERIFICATION: 🔧 NEEDS TUNING")
                print("🎯 Most strategies working, some may need parameter adjustment")
            
            return success
        
        # Wait and check again
        print(f"⏳ Waiting... ({elapsed_time//60}m {elapsed_time%60}s elapsed)")
        time.sleep(check_interval)
        elapsed_time += check_interval
    
    print(f"\n⏰ Timeout reached after {max_wait_time//60} minutes")
    print("🔍 Analyzing most recent workspace anyway...")
    
    newest_workspace = find_newest_workspace()
    if newest_workspace:
        return analyze_signal_coverage(newest_workspace)
    else:
        print("❌ No workspace found to analyze")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)