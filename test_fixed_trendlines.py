#!/usr/bin/env python3

"""Test the fixed trendline strategies by running a quick signal generation test."""

import sys
sys.path.append('/Users/daws/ADMF-PC/src')

import asyncio
from pathlib import Path
from core.coordinator.topology_runner import TopologyRunner

async def test_trendline_fixes():
    """Test that the trendline strategy fixes work by running signal generation."""
    
    # Create a minimal test configuration focusing on just the trendline strategies
    test_config = {
        'topology': {
            'data_sources': {
                'csv_source': {
                    'type': 'csv',
                    'path': '/Users/daws/ADMF-PC/data/SPY_1m_sample.csv',
                    'symbols': ['SPY'],
                    'timeframe': '1m'
                }
            },
            'strategies': {
                'trendline_breaks_test': {
                    'type': 'trendline_breaks',
                    'params': {
                        'pivot_lookback': 20,
                        'min_touches': 2,
                        'tolerance': 0.002
                    }
                },
                'trendline_bounces_test': {
                    'type': 'trendline_bounces', 
                    'params': {
                        'pivot_lookback': 20,
                        'min_touches': 3,
                        'tolerance': 0.002
                    }
                }
            },
            'connections': [
                {
                    'from': 'csv_source',
                    'to': ['trendline_breaks_test', 'trendline_bounces_test'],
                    'type': 'bar_data'
                }
            ]
        },
        'run': {
            'bars_limit': 100  # Just test a small number of bars
        }
    }
    
    print("Testing fixed trendline strategies...")
    
    # Create workspace
    workspace_dir = Path('/Users/daws/ADMF-PC/workspaces/trendline_test')
    workspace_dir.mkdir(exist_ok=True)
    
    try:
        # Run the topology
        runner = TopologyRunner(test_config, workspace_dir)
        await runner.run()
        
        # Check if signals were generated
        signal_files = list(workspace_dir.glob('**/*trendline*.parquet'))
        print(f"Found {len(signal_files)} trendline signal files:")
        for f in signal_files:
            print(f"  {f}")
        
        # Check analytics for signal counts
        analytics_db = workspace_dir / 'analytics.duckdb'
        if analytics_db.exists():
            import sqlite3
            conn = sqlite3.connect(str(analytics_db))
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT strategy_id, COUNT(*) FROM signals WHERE strategy_id LIKE '%trendline%' GROUP BY strategy_id")
                results = cursor.fetchall()
                
                print(f"\nSignal counts from analytics:")
                success_count = 0
                for strategy_id, count in results:
                    print(f"  {strategy_id}: {count} signals")
                    if count > 0:
                        success_count += 1
                
                print(f"\nResults: {success_count}/2 trendline strategies generated signals")
                return success_count >= 1  # At least one should work
                
            except Exception as e:
                print(f"Error querying analytics: {e}")
                return len(signal_files) > 0
            finally:
                conn.close()
        else:
            print("No analytics.db found")
            return len(signal_files) > 0
            
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_trendline_fixes())
    if result:
        print("\n✅ Trendline strategy fixes appear to be working!")
    else:
        print("\n❌ Trendline strategies may still have issues.")
    sys.exit(0 if result else 1)