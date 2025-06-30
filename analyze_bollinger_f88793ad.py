#!/usr/bin/env python3
"""
Analyze Bollinger Bands performance with various indicators
workspace: signal_generation_f88793ad
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def load_price_data():
    """Load the source price data."""
    # First, let's check what data files are available
    data_files = glob.glob("./data/SPY*.csv")
    if data_files:
        print(f"Found data files: {data_files}")
        # Use the most recent or appropriate one
        data_file = "./data/SPY_1m.csv"
    else:
        print("Warning: No SPY data files found in ./data/")
        return None
        
    try:
        prices = pd.read_csv(data_file)
        print(f"Loaded {len(prices)} bars from {data_file}")
        print(f"Columns: {list(prices.columns)}")
        return prices
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None

def calculate_indicators(prices):
    """Calculate technical indicators on price data."""
    # Ensure we have the right column names
    if 'Close' in prices.columns:
        close_col = 'Close'
        high_col = 'High'
        low_col = 'Low'
        volume_col = 'Volume'
    else:
        close_col = 'close'
        high_col = 'high' 
        low_col = 'low'
        volume_col = 'volume'
    
    # Don't need bar_idx - the index IS the bar_idx
    
    # Moving averages
    prices['sma_200'] = prices[close_col].rolling(200).mean()
    prices['sma_50'] = prices[close_col].rolling(50).mean()
    prices['sma_20'] = prices[close_col].rolling(20).mean()
    
    # Calculate VWAP ourselves since the data has NaN values
    prices['pv'] = prices[close_col] * prices[volume_col]
    prices['cumulative_pv'] = prices['pv'].cumsum()
    prices['cumulative_volume'] = prices[volume_col].cumsum()
    prices['calc_vwap'] = prices['cumulative_pv'] / prices['cumulative_volume']
    
    # Use calculated VWAP
    prices['vwap'] = prices['calc_vwap']
    
    # ATR for volatility
    prices['hl'] = prices[high_col] - prices[low_col]
    prices['hc'] = abs(prices[high_col] - prices[close_col].shift(1))
    prices['lc'] = abs(prices[low_col] - prices[close_col].shift(1))
    prices['tr'] = prices[['hl', 'hc', 'lc']].max(axis=1)
    prices['atr_14'] = prices['tr'].rolling(14).mean()
    prices['atr_pct'] = prices['atr_14'] / prices[close_col] * 100
    
    # Bollinger Bands
    prices['bb_middle'] = prices[close_col].rolling(20).mean()
    prices['bb_std'] = prices[close_col].rolling(20).std()
    prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
    prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
    prices['bb_width'] = (prices['bb_upper'] - prices['bb_lower']) / prices['bb_middle'] * 100
    
    # Price position within bands
    prices['bb_position'] = (prices[close_col] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])
    
    # Volume metrics
    prices['volume_sma'] = prices[volume_col].rolling(20).mean()
    prices['volume_ratio'] = prices[volume_col] / prices['volume_sma']
    
    # Rename close column for consistency
    prices['close'] = prices[close_col]
    
    return prices

def analyze_signal_quality(signal_file):
    """Analyze the quality of signals from a single file."""
    signals = pd.read_parquet(signal_file)
    
    print(f"\nAnalyzing: {signal_file.name}")
    print(f"Total signals: {len(signals)}")
    
    # Check for rapid signal changes
    rapid_changes = 0
    one_bar_trades = 0
    
    for i in range(1, len(signals)):
        bars_between = signals.iloc[i]['idx'] - signals.iloc[i-1]['idx']
        
        if bars_between == 1:
            # Check if it's a complete trade (entry then exit)
            if signals.iloc[i-1]['val'] != 0 and signals.iloc[i]['val'] == 0:
                one_bar_trades += 1
            rapid_changes += 1
    
    print(f"Rapid changes (1 bar apart): {rapid_changes}")
    print(f"One-bar trades: {one_bar_trades}")
    
    return signals

def extract_trades_with_context(signals, prices):
    """Extract trades and add context from price data."""
    
    trades = []
    entry_idx = None
    entry_signal = None
    
    for _, row in signals.iterrows():
        signal = row['val']
        bar_idx = row['idx']
        
        if entry_idx is None and signal != 0:
            # Entry - capture context
            # Use iloc since bar_idx is the actual row index
            if bar_idx >= len(prices):
                continue
                
            entry_context = prices.iloc[bar_idx]
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
            
        elif entry_idx is not None and (signal == 0 or signal != entry_signal):
            # Exit - calculate trade metrics
            # Use iloc since bar_idx is the actual row index
            if bar_idx >= len(prices):
                continue
                
            exit_context = prices.iloc[bar_idx]
            
            # PnL calculation
            if entry_signal > 0:
                pnl_pct = (row['px'] - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - row['px']) / entry_price * 100
            
            # Entry context - use iloc
            entry_price_row = prices.iloc[entry_idx]
            
            # Debug first trade
            if len(trades) == 0:
                print(f"\nDEBUG First Trade:")
                print(f"Entry idx: {entry_idx}, Exit idx: {bar_idx}")
                print(f"Entry SMA200: {entry_price_row['sma_200']}")
                print(f"Entry VWAP: {entry_price_row['vwap']}")
                print(f"Entry price: {entry_price}")
                
            trades.append({
                'entry_bar': entry_idx,
                'exit_bar': bar_idx,
                'duration': bar_idx - entry_idx,
                'signal_type': 'long' if entry_signal > 0 else 'short',
                'pnl_pct': pnl_pct,
                'entry_price': entry_price,
                'exit_price': row['px'],
                
                # Entry context
                'entry_above_sma200': entry_price > entry_price_row['sma_200'] if pd.notna(entry_price_row['sma_200']) else None,
                'entry_above_vwap': entry_price > entry_price_row['vwap'] if pd.notna(entry_price_row['vwap']) else None,
                'entry_atr_pct': entry_price_row['atr_pct'] if pd.notna(entry_price_row['atr_pct']) else None,
                'entry_bb_position': entry_price_row['bb_position'] if pd.notna(entry_price_row['bb_position']) else None,
                'entry_bb_width': entry_price_row['bb_width'] if pd.notna(entry_price_row['bb_width']) else None,
                'entry_volume_ratio': entry_price_row['volume_ratio'] if pd.notna(entry_price_row['volume_ratio']) else None,
            })
            
            # Check for re-entry
            if signal != 0:
                entry_idx = bar_idx
                entry_signal = signal
                entry_price = row['px']
            else:
                entry_idx = None
                
    return pd.DataFrame(trades)

def analyze_by_regime(trades_df):
    """Analyze performance by different regimes."""
    
    print("\n" + "="*60)
    print("REGIME ANALYSIS")
    print("="*60)
    
    # Remove trades with None values for analysis
    valid_trades = trades_df.dropna()
    
    if len(valid_trades) == 0:
        print("No valid trades with indicator data")
        return
    
    # Trend regime (SMA 200)
    if 'entry_above_sma200' in valid_trades.columns:
        print("\nBy SMA 200 Position:")
        above_sma = valid_trades[valid_trades['entry_above_sma200']]
        below_sma = valid_trades[~valid_trades['entry_above_sma200']]
        
        if len(above_sma) > 0:
            print(f"Above SMA 200: {len(above_sma)} trades, {above_sma['pnl_pct'].mean():.3f}% avg, "
                  f"{(above_sma['pnl_pct'] > 0).mean():.1%} win rate")
            print(f"  Net return: {above_sma['pnl_pct'].sum() - len(above_sma) * 0.01:.2f}%")
        
        if len(below_sma) > 0:
            print(f"Below SMA 200: {len(below_sma)} trades, {below_sma['pnl_pct'].mean():.3f}% avg, "
                  f"{(below_sma['pnl_pct'] > 0).mean():.1%} win rate")
            print(f"  Net return: {below_sma['pnl_pct'].sum() - len(below_sma) * 0.01:.2f}%")
    
    # VWAP position
    if 'entry_above_vwap' in valid_trades.columns:
        print("\nBy VWAP Position:")
        above_vwap = valid_trades[valid_trades['entry_above_vwap']]
        below_vwap = valid_trades[~valid_trades['entry_above_vwap']]
        
        if len(above_vwap) > 0:
            print(f"Above VWAP: {len(above_vwap)} trades, {above_vwap['pnl_pct'].mean():.3f}% avg, "
                  f"{(above_vwap['pnl_pct'] > 0).mean():.1%} win rate")
            print(f"  Net return: {above_vwap['pnl_pct'].sum() - len(above_vwap) * 0.01:.2f}%")
        
        if len(below_vwap) > 0:
            print(f"Below VWAP: {len(below_vwap)} trades, {below_vwap['pnl_pct'].mean():.3f}% avg, "
                  f"{(below_vwap['pnl_pct'] > 0).mean():.1%} win rate")
            print(f"  Net return: {below_vwap['pnl_pct'].sum() - len(below_vwap) * 0.01:.2f}%")
    
    # Volatility analysis
    if 'entry_atr_pct' in valid_trades.columns and valid_trades['entry_atr_pct'].notna().sum() > 0:
        print("\nBy Volatility (ATR %):")
        try:
            valid_trades['atr_quartile'] = pd.qcut(valid_trades['entry_atr_pct'].dropna(), 
                                                  q=4, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4(High)'])
            
            for quartile in ['Q1(Low)', 'Q2', 'Q3', 'Q4(High)']:
                q_trades = valid_trades[valid_trades['atr_quartile'] == quartile]
                if len(q_trades) > 0:
                    net_return = q_trades['pnl_pct'].sum() - len(q_trades) * 0.01
                    print(f"{quartile}: {len(q_trades)} trades, {q_trades['pnl_pct'].mean():.3f}% avg, "
                          f"{(q_trades['pnl_pct'] > 0).mean():.1%} win rate, {net_return:.2f}% net")
        except:
            print("Could not create ATR quartiles")
    
    # BB position analysis
    if 'entry_bb_position' in valid_trades.columns and valid_trades['entry_bb_position'].notna().sum() > 0:
        print("\nBy Bollinger Band Position at Entry:")
        try:
            valid_trades['bb_zone'] = pd.cut(valid_trades['entry_bb_position'].dropna(), 
                                           bins=[-np.inf, 0.2, 0.8, np.inf],
                                           labels=['Near Lower', 'Middle', 'Near Upper'])
            
            for zone in ['Near Lower', 'Middle', 'Near Upper']:
                z_trades = valid_trades[valid_trades['bb_zone'] == zone]
                if len(z_trades) > 0:
                    net_return = z_trades['pnl_pct'].sum() - len(z_trades) * 0.01
                    print(f"{zone}: {len(z_trades)} trades, {z_trades['pnl_pct'].mean():.3f}% avg, "
                          f"{(z_trades['pnl_pct'] > 0).mean():.1%} win rate, {net_return:.2f}% net")
        except:
            print("Could not create BB zones")
    
    # Duration analysis by regime
    print("\nDuration Analysis:")
    duration_buckets = [(1, 1), (2, 5), (6, 10), (11, 30), (31, 100), (101, 1000)]
    for low, high in duration_buckets:
        bucket_trades = valid_trades[(valid_trades['duration'] >= low) & (valid_trades['duration'] <= high)]
        if len(bucket_trades) > 0:
            net_return = bucket_trades['pnl_pct'].sum() - len(bucket_trades) * 0.01
            print(f"{low:3d}-{high:3d} bars: {len(bucket_trades):4d} trades, "
                  f"{bucket_trades['pnl_pct'].mean():6.3f}% avg, "
                  f"{(bucket_trades['pnl_pct'] > 0).mean():.1%} win, {net_return:6.2f}% net")

def main():
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99")
    
    # Find Bollinger Bands signal files
    bb_path = workspace_path / "traces" / "SPY_1m" / "signals" / "bollinger_bands"
    bb_files = list(bb_path.glob("*.parquet")) if bb_path.exists() else []
    print(f"Found {len(bb_files)} Bollinger Bands files")
    
    # Load price data
    prices = load_price_data()
    if prices is None:
        print("Could not load price data")
        return
    
    # Calculate indicators
    print("\nCalculating indicators...")
    prices = calculate_indicators(prices)
    
    # Analyze each BB strategy
    all_trades = []
    
    for signal_file in bb_files[:5]:  # Analyze first 5 for now
        signals = analyze_signal_quality(signal_file)
        
        # Extract trades with context
        trades_df = extract_trades_with_context(signals, prices)
        
        if len(trades_df) > 0:
            print(f"\nExtracted {len(trades_df)} trades")
            trades_df['strategy_file'] = signal_file.stem
            all_trades.append(trades_df)
            
            # Basic performance
            print(f"Win Rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
            print(f"Avg PnL: {trades_df['pnl_pct'].mean():.3f}%")
            print(f"Total Return: {trades_df['pnl_pct'].sum():.2f}%")
            print(f"Net Return (1bp cost): {trades_df['pnl_pct'].sum() - len(trades_df) * 0.01:.2f}%")
            
            # Detailed regime analysis
            analyze_by_regime(trades_df)
    
    # Combined analysis
    if all_trades:
        combined_df = pd.concat(all_trades, ignore_index=True)
        print("\n" + "="*60)
        print("COMBINED ANALYSIS - ALL STRATEGIES")
        print("="*60)
        print(f"Total trades analyzed: {len(combined_df)}")
        analyze_by_regime(combined_df)

if __name__ == "__main__":
    main()