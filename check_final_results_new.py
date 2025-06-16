#!/usr/bin/env python3

"""Check the final results to see if our trendline strategy fixes worked."""

import sys
sys.path.append('/Users/daws/ADMF-PC/src')

import sqlite3
from pathlib import Path

def check_final_strategy_results():
    """Check all strategies and see if the trendline ones are now working."""
    
    # Find the latest workspace
    workspaces = list(Path('/Users/daws/ADMF-PC/workspaces').glob('20*'))
    if not workspaces:
        print("No recent workspaces found")
        return
    
    latest_workspace = max(workspaces, key=lambda p: p.stat().st_mtime)
    print(f"Checking workspace: {latest_workspace}")
    
    analytics_db = latest_workspace / 'analytics.duckdb'
    if not analytics_db.exists():
        print("No analytics.duckdb found")
        return
    
    # Connect and query strategy results
    conn = sqlite3.connect(str(analytics_db))
    cursor = conn.cursor()
    
    try:
        # Get strategy execution results
        cursor.execute("""
            SELECT strategy_name, 
                   COUNT(*) as signal_count,
                   COUNT(DISTINCT strategy_id) as unique_variants,
                   MIN(signal_value) as min_signal,
                   MAX(signal_value) as max_signal
            FROM signals 
            GROUP BY strategy_name 
            ORDER BY strategy_name
        """)
        
        all_strategies = cursor.fetchall()
        
        print(f"\nFound {len(all_strategies)} strategy types with signals:")
        print("="*80)
        
        trendline_strategies = []
        working_strategies = 0
        total_strategies = len(all_strategies)
        
        for strategy_name, signal_count, variants, min_sig, max_sig in all_strategies:
            status = "✓ WORKING" if signal_count > 0 else "✗ FAILED"
            print(f"{status:12} {strategy_name:35} {signal_count:6} signals ({variants:2} variants) range:[{min_sig:2},{max_sig:2}]")
            
            if signal_count > 0:
                working_strategies += 1
            
            if 'trendline' in strategy_name:
                trendline_strategies.append((strategy_name, signal_count, status))
        
        print("="*80)
        print(f"Overall: {working_strategies}/{total_strategies} strategies working ({working_strategies/total_strategies*100:.1f}%)")
        
        # Focus on trendline strategies
        print(f"\nTRENDLINE STRATEGY ANALYSIS:")
        print("="*50)
        
        if trendline_strategies:
            working_trendlines = sum(1 for _, count, _ in trendline_strategies if count > 0)
            total_trendlines = len(trendline_strategies)
            
            for strategy_name, signal_count, status in trendline_strategies:
                print(f"{status:12} {strategy_name:25} {signal_count:6} signals")
            
            print(f"\nTrendline Results: {working_trendlines}/{total_trendlines} working")
            
            if working_trendlines == total_trendlines:
                print("🎉 ALL TRENDLINE STRATEGIES FIXED!")
                return True
            elif working_trendlines > 0:
                print("✅ Some trendline strategies now working!")
                return True
            else:
                print("❌ Trendline strategies still not working")
                return False
        else:
            print("No trendline strategies found in results")
            
            # Check if trendline strategies were even attempted
            cursor.execute("SELECT DISTINCT strategy_name FROM signals WHERE strategy_name LIKE '%trend%'")
            trend_related = cursor.fetchall()
            if trend_related:
                print("Found trend-related strategies:")
                for name, in trend_related:
                    print(f"  {name}")
            else:
                print("No trend-related strategies found at all")
            
            return False
    
    except Exception as e:
        print(f"Error querying results: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = check_final_strategy_results()
    print(f"\nFinal status: {'SUCCESS' if success else 'NEEDS MORE WORK'}")
    sys.exit(0 if success else 1)