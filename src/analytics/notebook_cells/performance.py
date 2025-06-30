"""
Performance calculation cells for notebooks.

These functions return complete cell code that can be executed in notebooks.
"""

def sharpe_calculation_cell():
    """Cell for calculating Sharpe ratios"""
    return """
# Calculate Sharpe ratio for strategies
def calculate_sharpe(returns, periods_per_year=252*78):
    '''Calculate annualized Sharpe ratio'''
    if returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)

# Apply to all strategies
if 'performance_df' in locals():
    print("Recalculating Sharpe ratios...")
    for idx, row in performance_df.iterrows():
        strategy_returns = get_strategy_returns(row['strategy_hash'])
        performance_df.loc[idx, 'sharpe_ratio'] = calculate_sharpe(strategy_returns)
    
    print(f"Top 5 by Sharpe:")
    print(performance_df.nlargest(5, 'sharpe_ratio')[['strategy_type', 'sharpe_ratio', 'total_return']])
"""


def drawdown_analysis_cell():
    """Cell for drawdown analysis"""
    return """
# Analyze drawdowns
def calculate_drawdowns(equity_curve):
    '''Calculate drawdown series and statistics'''
    cummax = equity_curve.expanding().max()
    drawdown = (equity_curve / cummax - 1)
    
    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_starts = (~in_drawdown.shift(1).fillna(False)) & in_drawdown
    drawdown_ends = in_drawdown.shift(1).fillna(False) & (~in_drawdown)
    
    return {
        'max_drawdown': drawdown.min(),
        'avg_drawdown': drawdown[drawdown < 0].mean(),
        'drawdown_periods': drawdown_starts.sum(),
        'longest_drawdown': (drawdown < 0).astype(int).groupby((drawdown >= 0).cumsum()).cumsum().max()
    }

# Analyze drawdowns for top strategies
if 'top_performers' in locals():
    print("\\nDrawdown Analysis:")
    print("-" * 50)
    
    for idx, strategy in top_performers.head(5).iterrows():
        equity = get_equity_curve(strategy['strategy_hash'])
        dd_stats = calculate_drawdowns(equity)
        
        print(f"\\n{strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        print(f"  Max Drawdown: {dd_stats['max_drawdown']:.1%}")
        print(f"  Avg Drawdown: {dd_stats['avg_drawdown']:.1%}")
        print(f"  Drawdown Periods: {dd_stats['drawdown_periods']}")
        print(f"  Longest Drawdown: {dd_stats['longest_drawdown']} bars")
"""


def win_rate_analysis_cell():
    """Cell for win rate and trade analysis"""
    return """
# Analyze win rates and trade statistics
def analyze_trades(strategy_hash, signals_df, market_data):
    '''Analyze individual trades'''
    # Merge signals with market data
    df = market_data.merge(signals_df, on='timestamp', how='left')
    df['signal'] = df['signal'].fillna(method='ffill').fillna(0)
    
    # Identify trade entry/exit points
    df['position_change'] = df['signal'].diff()
    entries = df[df['position_change'] != 0].copy()
    
    if len(entries) < 2:
        return None
    
    # Calculate trade returns
    trades = []
    for i in range(len(entries)-1):
        entry = entries.iloc[i]
        exit = entries.iloc[i+1]
        
        if entry['signal'] != 0:  # Not an exit
            # Correctly calculate returns for long and short positions
            if entry['signal'] > 0:  # Long position
                trade_return = (exit['close'] - entry['close']) / entry['close']
            else:  # Short position
                trade_return = (entry['close'] - exit['close']) / entry['close']
            
            trades.append({
                'entry_time': entry['timestamp'],
                'exit_time': exit['timestamp'],
                'direction': 'long' if entry['signal'] > 0 else 'short',
                'return': trade_return,
                'duration_bars': exit.name - entry.name
            })
    
    trades_df = pd.DataFrame(trades)
    
    return {
        'total_trades': len(trades_df),
        'win_rate': (trades_df['return'] > 0).mean(),
        'avg_win': trades_df[trades_df['return'] > 0]['return'].mean(),
        'avg_loss': trades_df[trades_df['return'] < 0]['return'].mean(),
        'profit_factor': abs(trades_df[trades_df['return'] > 0]['return'].sum() / 
                            trades_df[trades_df['return'] < 0]['return'].sum()),
        'avg_duration': trades_df['duration_bars'].mean(),
        'trades_df': trades_df
    }

# Run trade analysis
print("\\nTrade Analysis for Top Strategies:")
print("=" * 70)

for idx, strategy in top_performers.head(5).iterrows():
    trade_stats = analyze_trades(strategy['strategy_hash'], 
                                get_signals(strategy['strategy_hash']), 
                                market_data)
    
    if trade_stats:
        print(f"\\n{strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        print(f"  Total Trades: {trade_stats['total_trades']}")
        print(f"  Win Rate: {trade_stats['win_rate']:.1%}")
        print(f"  Avg Win: {trade_stats['avg_win']:.2%}")
        print(f"  Avg Loss: {trade_stats['avg_loss']:.2%}")
        print(f"  Profit Factor: {trade_stats['profit_factor']:.2f}")
        print(f"  Avg Duration: {trade_stats['avg_duration']:.0f} bars")
"""


def performance_attribution_cell():
    """Cell for performance attribution analysis"""
    return """
# Performance attribution - what drives returns?
def attribute_performance(strategy_data):
    '''Break down performance by various factors'''
    
    # Time-based attribution
    strategy_data['hour'] = strategy_data['timestamp'].dt.hour
    strategy_data['day'] = strategy_data['timestamp'].dt.dayofweek
    strategy_data['month'] = strategy_data['timestamp'].dt.month
    
    attributions = {}
    
    # By hour of day
    attributions['by_hour'] = strategy_data.groupby('hour')['strategy_returns'].agg(['mean', 'sum', 'count'])
    
    # By day of week
    attributions['by_day'] = strategy_data.groupby('day')['strategy_returns'].agg(['mean', 'sum', 'count'])
    
    # By market conditions
    if 'volatility_regime' in strategy_data.columns:
        attributions['by_volatility'] = strategy_data.groupby('volatility_regime')['strategy_returns'].agg(['mean', 'sum', 'count'])
    
    return attributions

# Run attribution for best strategy
if len(performance_df) > 0:
    best = performance_df.iloc[0]
    print(f"\\nPerformance Attribution for {best['strategy_type']} - {best['strategy_hash'][:8]}")
    print("-" * 60)
    
    # Get strategy data with returns
    strategy_data = get_strategy_data(best['strategy_hash'])
    attributions = attribute_performance(strategy_data)
    
    # Plot hourly attribution
    if 'by_hour' in attributions:
        plt.figure(figsize=(12, 4))
        attributions['by_hour']['mean'].plot(kind='bar')
        plt.title('Average Returns by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Average Return')
        plt.tight_layout()
        plt.show()
"""