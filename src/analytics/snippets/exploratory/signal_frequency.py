# Signal frequency analysis - understand trading patterns
# Edit these parameters as needed:
MIN_SIGNALS = 10  # Minimum signals to include
MIN_SHARPE = 1.0  # Minimum Sharpe ratio filter
TIMEFRAME = '1H'  # Aggregation timeframe for patterns

# Basic signal frequency query
signal_freq_sql = f"""
WITH signal_summary AS (
    SELECT 
        strategy_hash,
        COUNT(*) as total_signals,
        COUNT(DISTINCT DATE(ts)) as trading_days,
        MIN(ts) as first_signal,
        MAX(ts) as last_signal,
        SUM(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_signals,
        SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) as short_signals,
        -- Intraday distribution
        COUNT(DISTINCT DATE_TRUNC('hour', ts)) as active_hours
    FROM signals
    WHERE val != 0
    GROUP BY strategy_hash
    HAVING COUNT(*) >= {MIN_SIGNALS}
)
SELECT 
    ss.*,
    st.strategy_type,
    st.sharpe_ratio,
    st.total_return,
    -- Calculate rates
    ss.total_signals::FLOAT / NULLIF(ss.trading_days, 0) as signals_per_day,
    ss.total_signals::FLOAT / NULLIF(ss.active_hours, 0) as signals_per_hour,
    ss.long_signals::FLOAT / NULLIF(ss.total_signals, 0) as long_bias,
    -- Activity span
    EXTRACT(EPOCH FROM (last_signal - first_signal)) / 86400.0 as active_days
FROM signal_summary ss
JOIN strategies st ON ss.strategy_hash = st.strategy_hash
WHERE st.sharpe_ratio >= {MIN_SHARPE}
ORDER BY st.sharpe_ratio DESC
"""

print(f"Analyzing signal frequency (min_signals={MIN_SIGNALS}, min_sharpe={MIN_SHARPE})...")
freq_df = con.execute(signal_freq_sql).df()
print(f"Found {len(freq_df)} strategies")

# Summary statistics
print("\nSignal Frequency Summary:")
print("-" * 50)
print(f"Average signals per day: {freq_df['signals_per_day'].mean():.1f}")
print(f"Average long bias: {freq_df['long_bias'].mean():.1%}")
print(f"Most active strategy: {freq_df['signals_per_day'].max():.1f} signals/day")
print(f"Least active strategy: {freq_df['signals_per_day'].min():.1f} signals/day")

# Group by strategy type
type_summary = freq_df.groupby('strategy_type').agg({
    'signals_per_day': ['mean', 'std', 'min', 'max'],
    'long_bias': 'mean',
    'sharpe_ratio': 'mean'
}).round(2)
print("\nBy Strategy Type:")
print(type_summary)

# Visualize distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Signals per day distribution
axes[0].hist(freq_df['signals_per_day'], bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(freq_df['signals_per_day'].mean(), color='red', linestyle='--', 
                label=f'Mean: {freq_df["signals_per_day"].mean():.1f}')
axes[0].set_xlabel('Signals per Day')
axes[0].set_ylabel('Number of Strategies')
axes[0].set_title('Trading Frequency Distribution')
axes[0].legend()

# Long bias by strategy type
type_bias = freq_df.groupby('strategy_type')['long_bias'].mean().sort_values()
type_bias.plot(kind='barh', ax=axes[1])
axes[1].axvline(0.5, color='black', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Long Bias (0=all short, 1=all long)')
axes[1].set_title('Directional Bias by Strategy Type')

plt.tight_layout()
plt.show()

# Store results for further analysis
signal_frequency_df = freq_df