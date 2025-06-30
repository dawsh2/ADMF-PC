"""
Regime analysis cells for understanding strategy performance in different market conditions.
"""

def volatility_regime_cell():
    """Cell for analyzing performance by volatility regime"""
    return """
# Analyze performance by volatility regime
def calculate_volatility_regimes(market_data, lookback=20, quantiles=[0.33, 0.67]):
    '''Classify market into volatility regimes'''
    
    # Calculate rolling volatility
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(lookback).std() * np.sqrt(252 * 78)
    
    # Define regimes based on quantiles
    vol_low = market_data['volatility'].quantile(quantiles[0])
    vol_high = market_data['volatility'].quantile(quantiles[1])
    
    market_data['vol_regime'] = pd.cut(
        market_data['volatility'],
        bins=[-np.inf, vol_low, vol_high, np.inf],
        labels=['Low', 'Medium', 'High']
    )
    
    return market_data

def analyze_regime_performance(strategy_data, market_data_with_regimes):
    '''Analyze strategy performance in different regimes'''
    
    # Merge strategy data with regimes
    merged = strategy_data.merge(
        market_data_with_regimes[['vol_regime']], 
        left_index=True, 
        right_index=True
    )
    
    # Calculate performance by regime
    regime_stats = merged.groupby('vol_regime').agg({
        'strategy_returns': ['mean', 'std', 'count', 'sum'],
        'signal': lambda x: (x != 0).sum()  # Number of signals
    })
    
    regime_stats.columns = ['avg_return', 'volatility', 'num_bars', 'total_return', 'num_signals']
    regime_stats['sharpe'] = regime_stats['avg_return'] / regime_stats['volatility'] * np.sqrt(252 * 78)
    regime_stats['signals_per_day'] = regime_stats['num_signals'] / (regime_stats['num_bars'] / 78)
    
    return regime_stats

# Calculate regimes
market_with_regimes = calculate_volatility_regimes(market_data.copy())

print("\\nVolatility Regime Distribution:")
print(market_with_regimes['vol_regime'].value_counts().sort_index())
print(f"\\nAverage volatility by regime:")
for regime in ['Low', 'Medium', 'High']:
    avg_vol = market_with_regimes[market_with_regimes['vol_regime'] == regime]['volatility'].mean()
    print(f"  {regime}: {avg_vol:.1%}")

# Analyze top strategies by regime
if 'top_performers' in locals():
    print("\\n" + "="*60)
    print("REGIME PERFORMANCE ANALYSIS")
    print("="*60)
    
    for idx, strategy in top_performers.head(3).iterrows():
        print(f"\\n{strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        print("-" * 40)
        
        strategy_data = get_strategy_data(strategy['strategy_hash'])
        regime_stats = analyze_regime_performance(strategy_data, market_with_regimes)
        
        print(regime_stats.round(4))
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        regime_stats['sharpe'].plot(kind='bar', ax=ax1)
        ax1.set_title('Sharpe Ratio by Volatility Regime')
        ax1.set_xlabel('Volatility Regime')
        ax1.set_ylabel('Sharpe Ratio')
        
        regime_stats['signals_per_day'].plot(kind='bar', ax=ax2)
        ax2.set_title('Trading Frequency by Regime')
        ax2.set_xlabel('Volatility Regime')
        ax2.set_ylabel('Signals per Day')
        
        plt.suptitle(f"{strategy['strategy_type']} - Regime Analysis")
        plt.tight_layout()
        plt.show()
"""


def trend_regime_cell():
    """Cell for trend regime analysis"""
    return """
# Analyze performance in trending vs ranging markets
def calculate_trend_regimes(market_data, short_ma=20, long_ma=50, atr_period=14):
    '''Classify market into trend/range regimes'''
    
    # Calculate moving averages
    market_data['ma_short'] = market_data['close'].rolling(short_ma).mean()
    market_data['ma_long'] = market_data['close'].rolling(long_ma).mean()
    
    # Calculate trend strength
    market_data['trend_strength'] = (market_data['ma_short'] - market_data['ma_long']) / market_data['ma_long']
    
    # Calculate ATR for volatility-adjusted thresholds
    high = market_data['high']
    low = market_data['low']
    close = market_data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    market_data['atr'] = tr.rolling(atr_period).mean()
    
    # Normalized trend strength
    market_data['trend_strength_normalized'] = market_data['trend_strength'] / (market_data['atr'] / market_data['close'])
    
    # Define regimes
    trend_threshold = 0.5
    market_data['trend_regime'] = pd.cut(
        market_data['trend_strength_normalized'],
        bins=[-np.inf, -trend_threshold, trend_threshold, np.inf],
        labels=['Downtrend', 'Range', 'Uptrend']
    )
    
    return market_data

# Calculate trend regimes
market_with_trends = calculate_trend_regimes(market_data.copy())

# Visualize regime distribution over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Price with MAs
ax1.plot(market_data.index, market_data['close'], label='Close', alpha=0.7)
ax1.plot(market_data.index, market_with_trends['ma_short'], label='MA20', alpha=0.8)
ax1.plot(market_data.index, market_with_trends['ma_long'], label='MA50', alpha=0.8)
ax1.set_ylabel('Price')
ax1.legend()
ax1.set_title('Price and Moving Averages')

# Regime indicator
regime_colors = {'Uptrend': 'green', 'Range': 'yellow', 'Downtrend': 'red'}
for regime, color in regime_colors.items():
    mask = market_with_trends['trend_regime'] == regime
    ax2.fill_between(market_data.index, 0, 1, where=mask, 
                     color=color, alpha=0.3, label=regime)

ax2.set_ylabel('Regime')
ax2.set_ylim(0, 1)
ax2.legend()
ax2.set_title('Market Regime Classification')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.show()

# Analyze strategy performance by trend regime
print("\\nTrend Regime Distribution:")
print(market_with_trends['trend_regime'].value_counts(normalize=True).sort_index())
"""


def regime_transition_cell():
    """Cell for analyzing regime transitions"""
    return """
# Analyze strategy behavior during regime transitions
def analyze_regime_transitions(strategy_data, market_with_regimes, regime_column='vol_regime'):
    '''Analyze performance around regime changes'''
    
    # Identify regime changes
    regime_changes = market_with_regimes[regime_column].ne(market_with_regimes[regime_column].shift())
    transition_dates = market_with_regimes.index[regime_changes]
    
    # Analyze performance around transitions
    window_before = 10  # bars before transition
    window_after = 10   # bars after transition
    
    transition_performance = []
    
    for date in transition_dates[1:-1]:  # Skip first and last
        try:
            # Get position in index
            pos = market_with_regimes.index.get_loc(date)
            
            # Get strategy returns around transition
            returns_before = strategy_data.iloc[pos-window_before:pos]['strategy_returns'].sum()
            returns_after = strategy_data.iloc[pos:pos+window_after]['strategy_returns'].sum()
            
            # Get regime info
            regime_from = market_with_regimes[regime_column].iloc[pos-1]
            regime_to = market_with_regimes[regime_column].iloc[pos]
            
            transition_performance.append({
                'date': date,
                'from_regime': regime_from,
                'to_regime': regime_to,
                'returns_before': returns_before,
                'returns_after': returns_after,
                'total_impact': returns_before + returns_after
            })
            
        except:
            continue
    
    transitions_df = pd.DataFrame(transition_performance)
    
    # Aggregate by transition type
    transition_summary = transitions_df.groupby(['from_regime', 'to_regime']).agg({
        'total_impact': ['mean', 'std', 'count'],
        'returns_before': 'mean',
        'returns_after': 'mean'
    }).round(4)
    
    return transitions_df, transition_summary

# Analyze transitions for best strategy
if len(performance_df) > 0 and 'market_with_regimes' in locals():
    best = performance_df.iloc[0]
    strategy_data = get_strategy_data(best['strategy_hash'])
    
    transitions, summary = analyze_regime_transitions(strategy_data, market_with_regimes)
    
    print(f"\\nRegime Transition Analysis for {best['strategy_type']}")
    print("="*60)
    print(summary)
    
    # Visualize transition impacts
    plt.figure(figsize=(10, 6))
    transition_pivot = transitions.pivot_table(
        index='from_regime', 
        columns='to_regime', 
        values='total_impact',
        aggfunc='mean'
    )
    
    sns.heatmap(transition_pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0)
    plt.title('Average Returns During Regime Transitions (±10 bars)')
    plt.tight_layout()
    plt.show()
"""