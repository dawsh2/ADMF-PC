#!/usr/bin/env python3
"""
Analyze the win/loss profile to understand how low win rate produces high Sharpe.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_trade_profile(df):
    """Analyze individual trade characteristics."""
    
    if df.empty:
        return
    
    # Rename columns
    df = df.rename(columns={
        'idx': 'bar_idx',
        'px': 'price',
        'val': 'signal_value'
    })
    
    # Calculate trades
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    
    print("Calculating individual trades...")
    
    for idx, row in df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        if current_position == 0:
            # No position, check if we should open one
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
        else:
            # We have a position
            if signal == 0 or signal != current_position:
                # Close position
                if entry_price > 0 and price > 0:
                    # Calculate return
                    if current_position == 1:  # Long
                        trade_return = (price - entry_price) / entry_price
                    else:  # Short
                        trade_return = (entry_price - price) / entry_price
                    
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'return_pct': trade_return,
                        'bars_held': bar_idx - entry_bar_idx,
                        'is_winner': trade_return > 0
                    })
                
                # Reset or flip position
                current_position = 0
                entry_price = None
                entry_bar_idx = None
                
                if signal != 0:  # Signal flip
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
    
    if not trades:
        print("No trades found!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Basic stats
    total_trades = len(trades_df)
    winners = trades_df[trades_df['is_winner']]
    losers = trades_df[~trades_df['is_winner']]
    
    win_rate = len(winners) / total_trades
    
    print(f"\n{'='*70}")
    print("TRADE PROFILE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"Total trades: {total_trades}")
    print(f"Winners: {len(winners)} ({win_rate:.1%})")
    print(f"Losers: {len(losers)} ({(1-win_rate):.1%})")
    
    # Win/Loss magnitudes
    if len(winners) > 0:
        avg_win = winners['return_pct'].mean()
        median_win = winners['return_pct'].median()
        max_win = winners['return_pct'].max()
        print(f"\nWINNING TRADES:")
        print(f"  Average win: {avg_win:.3%}")
        print(f"  Median win: {median_win:.3%}")
        print(f"  Largest win: {max_win:.3%}")
    
    if len(losers) > 0:
        avg_loss = losers['return_pct'].mean()
        median_loss = losers['return_pct'].median()
        max_loss = losers['return_pct'].min()  # Most negative
        print(f"\nLOSING TRADES:")
        print(f"  Average loss: {avg_loss:.3%}")
        print(f"  Median loss: {median_loss:.3%}")
        print(f"  Largest loss: {max_loss:.3%}")
    
    # Win/Loss ratio
    if len(winners) > 0 and len(losers) > 0:
        win_loss_ratio = abs(winners['return_pct'].mean() / losers['return_pct'].mean())
        print(f"\nWIN/LOSS RATIO:")
        print(f"  Avg win / Avg loss: {win_loss_ratio:.2f}x")
    
    # Expected value per trade
    expected_value = trades_df['return_pct'].mean()
    print(f"\nEXPECTED VALUE:")
    print(f"  Per trade: {expected_value:.4%}")
    
    # Return distribution analysis
    print(f"\nRETURN DISTRIBUTION:")
    print(f"  Standard deviation: {trades_df['return_pct'].std():.3%}")
    print(f"  Skewness: {trades_df['return_pct'].skew():.2f}")
    print(f"  Kurtosis: {trades_df['return_pct'].kurtosis():.2f}")
    
    # Percentile analysis
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nRETURN PERCENTILES:")
    for p in percentiles:
        pct_val = np.percentile(trades_df['return_pct'], p)
        print(f"  {p:2d}th percentile: {pct_val:.3%}")
    
    # Trade duration analysis
    print(f"\nTRADE DURATION:")
    print(f"  Average hold: {trades_df['bars_held'].mean():.1f} bars")
    print(f"  Median hold: {trades_df['bars_held'].median():.1f} bars")
    
    # Long vs Short performance
    long_trades = trades_df[trades_df['signal'] == 1]
    short_trades = trades_df[trades_df['signal'] == -1]
    
    if len(long_trades) > 0:
        long_win_rate = (long_trades['return_pct'] > 0).mean()
        long_avg_return = long_trades['return_pct'].mean()
        print(f"\nLONG TRADES:")
        print(f"  Count: {len(long_trades)}")
        print(f"  Win rate: {long_win_rate:.1%}")
        print(f"  Avg return: {long_avg_return:.3%}")
    
    if len(short_trades) > 0:
        short_win_rate = (short_trades['return_pct'] > 0).mean()
        short_avg_return = short_trades['return_pct'].mean()
        print(f"\nSHORT TRADES:")
        print(f"  Count: {len(short_trades)}")
        print(f"  Win rate: {short_win_rate:.1%}")
        print(f"  Avg return: {short_avg_return:.3%}")
    
    # High Sharpe explanation
    print(f"\n{'='*70}")
    print("HIGH SHARPE RATIO EXPLANATION")
    print(f"{'='*70}")
    
    if len(winners) > 0 and len(losers) > 0:
        win_loss_ratio = abs(winners['return_pct'].mean() / losers['return_pct'].mean())
        
        print(f"Mathematical breakdown:")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Loss rate: {(1-win_rate):.1%}")
        print(f"  Avg win: {winners['return_pct'].mean():.3%}")
        print(f"  Avg loss: {losers['return_pct'].mean():.3%}")
        print(f"  Win/Loss ratio: {win_loss_ratio:.2f}x")
        
        # Kelly Criterion check
        win_prob = win_rate
        loss_prob = 1 - win_rate
        avg_win_pct = winners['return_pct'].mean()
        avg_loss_pct = abs(losers['return_pct'].mean())
        
        if avg_loss_pct > 0:
            kelly_fraction = (win_prob * avg_win_pct - loss_prob * avg_loss_pct) / avg_loss_pct
            print(f"  Kelly fraction: {kelly_fraction:.3f}")
        
        # Expected return calculation
        expected_return = win_prob * avg_win_pct + loss_prob * losers['return_pct'].mean()
        print(f"  Expected return: {expected_return:.4%}")
        
        print(f"\nKey insight:")
        if win_loss_ratio > (1-win_rate)/win_rate:
            print(f"  The average win ({winners['return_pct'].mean():.3%}) is large enough")
            print(f"  to overcome the lower win rate ({win_rate:.1%})")
            print(f"  This creates positive expected value with low volatility")
        else:
            print(f"  Something unusual is happening - need deeper analysis")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_win_loss_profile.py <parquet_file_path>")
        sys.exit(1)
    
    signal_file = Path(sys.argv[1])
    if not signal_file.exists():
        print(f"File not found: {signal_file}")
        sys.exit(1)
    
    print(f"Analyzing trade profile for: {signal_file}")
    df = pd.read_parquet(signal_file)
    analyze_trade_profile(df)

if __name__ == "__main__":
    main()