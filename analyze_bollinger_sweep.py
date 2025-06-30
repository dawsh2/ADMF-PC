#!/usr/bin/env python3
"""
Analyze Bollinger Bands parameter sweep results to find best performers.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def analyze_bollinger_sweep():
    """Analyze the parameter sweep results."""
    
    print("Bollinger Bands Parameter Sweep Analysis")
    print("=" * 50)
    
    # Check what's in the bollinger results directory
    bollinger_path = Path("config/bollinger/results/latest")
    
    if not bollinger_path.exists():
        print(f"❌ Path not found: {bollinger_path}")
        # Try to find the actual results
        bollinger_base = Path("config/bollinger/results")
        if bollinger_base.exists():
            dirs = sorted([d for d in bollinger_base.iterdir() if d.is_dir()])
            if dirs:
                print(f"\nFound {len(dirs)} result directories:")
                for d in dirs[-5:]:  # Show last 5
                    print(f"  - {d.name}")
                bollinger_path = dirs[-1]  # Use most recent
                print(f"\nUsing most recent: {bollinger_path.name}")
        else:
            print("No bollinger results found")
            return
    
    # Load metadata
    metadata_path = bollinger_path / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nWorkspace: {metadata.get('workspace_path', 'Unknown')}")
    print(f"Total components: {len(metadata.get('components', {}))}")
    
    # Analyze each strategy's performance
    results = []
    
    for comp_id, comp_data in metadata.get('components', {}).items():
        if comp_data['component_type'] != 'strategy':
            continue
            
        # Extract parameters from component ID or metadata
        params = comp_data.get('parameters', {})
        
        # Try to load the signal file to calculate performance
        signal_file = bollinger_path / comp_data['signal_file_path']
        if signal_file.exists():
            try:
                signals = pd.read_parquet(signal_file)
                
                # Basic performance metrics
                total_signals = len(signals)
                non_zero_signals = len(signals[signals['val'] != 0])
                
                # Extract strategy number from component ID
                # Format like: SPY_5m_compiled_strategy_123
                if 'compiled_strategy_' in comp_id:
                    strategy_num = int(comp_id.split('compiled_strategy_')[1])
                else:
                    strategy_num = -1
                
                result = {
                    'component_id': comp_id,
                    'strategy_num': strategy_num,
                    'total_signals': total_signals,
                    'non_zero_signals': non_zero_signals,
                    'signal_frequency': non_zero_signals / comp_data.get('total_bars', 1),
                    'parameters': params
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error loading {signal_file}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\n❌ No strategy results found")
        return
    
    print(f"\nAnalyzed {len(df)} strategies")
    
    # Try to decode parameters from strategy numbers if config is available
    config_path = Path("config/bollinger/config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if this was a parameter sweep
        if 'strategy' in config and isinstance(config['strategy'], list):
            strategy_config = config['strategy'][0]  # First strategy
            
            if 'bollinger_bands' in strategy_config:
                bb_config = strategy_config['bollinger_bands']
                
                # Check for parameter lists
                periods = bb_config.get('period', [])
                std_devs = bb_config.get('std_dev', [])
                
                if isinstance(periods, list) and isinstance(std_devs, list):
                    print(f"\nParameter grid:")
                    print(f"  Periods: {periods}")
                    print(f"  Std devs: {std_devs}")
                    
                    # Decode strategy parameters
                    for idx, row in df.iterrows():
                        strategy_num = row['strategy_num']
                        if strategy_num >= 0:
                            # Calculate which parameter combination this is
                            std_idx = strategy_num % len(std_devs)
                            period_idx = strategy_num // len(std_devs)
                            
                            if period_idx < len(periods):
                                df.at[idx, 'period'] = periods[period_idx]
                                df.at[idx, 'std_dev'] = std_devs[std_idx]
    
    # Sort by signal frequency or other metrics
    df_sorted = df.sort_values('signal_frequency', ascending=False)
    
    print("\nTop 10 strategies by signal frequency:")
    print("-" * 60)
    
    for idx, row in df_sorted.head(10).iterrows():
        period = row.get('period', '?')
        std_dev = row.get('std_dev', '?')
        print(f"Strategy {row['strategy_num']}: period={period}, std_dev={std_dev}")
        print(f"  Signal frequency: {row['signal_frequency']:.2%}")
        print(f"  Non-zero signals: {row['non_zero_signals']}")
        print()
    
    # Find the specific combination mentioned (period=11, std_dev=2.0)
    if 'period' in df.columns and 'std_dev' in df.columns:
        target = df[(df['period'] == 11) & (df['std_dev'] == 2.0)]
        if not target.empty:
            print("\nTarget strategy (period=11, std_dev=2.0):")
            print(f"  Strategy number: {target.iloc[0]['strategy_num']}")
            print(f"  Signal frequency: {target.iloc[0]['signal_frequency']:.2%}")
            print(f"  Non-zero signals: {target.iloc[0]['non_zero_signals']}")
        else:
            print("\n⚠️ Could not find period=11, std_dev=2.0 in results")
    
    # Save detailed results
    output_path = Path("bollinger_analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    analyze_bollinger_sweep()