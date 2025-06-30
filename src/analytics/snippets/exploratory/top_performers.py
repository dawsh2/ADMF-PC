# Find and analyze top performing strategies
# Edit these parameters before running:
MIN_SHARPE = 1.5          # Minimum Sharpe ratio
MIN_TRADES = 20           # Minimum number of trades
MAX_DRAWDOWN_LIMIT = -0.3 # Maximum acceptable drawdown (negative)
TOP_N = 20                # Number of top strategies to analyze

# Load the top strategies query
with open('src/analytics/queries/top_strategies.sql', 'r') as f:
    top_strategies_sql = f.read()

# Execute with parameters
print(f"Finding top performers (sharpe>{MIN_SHARPE}, trades>{MIN_TRADES})...")
top_df = con.execute(top_strategies_sql.format(
    min_sharpe=MIN_SHARPE,
    min_trades=MIN_TRADES,
    max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
    limit=TOP_N
)).df()

print(f"Found {len(top_df)} strategies meeting criteria")

# Display top 10
print("\nTop 10 Strategies:")
print("-" * 80)
display_cols = ['strategy_type', 'sharpe_ratio', 'total_return', 'max_drawdown', 
                'win_rate', 'total_trades', 'return_drawdown_ratio']
print(top_df[display_cols].head(10).to_string(index=False))

# Analyze by strategy type
print("\nPerformance by Strategy Type:")
type_stats = top_df.groupby('strategy_type').agg({
    'sharpe_ratio': ['count', 'mean', 'max'],
    'total_return': 'mean',
    'max_drawdown': 'mean',
    'win_rate': 'mean'
}).round(3)
print(type_stats)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sharpe ratio distribution
axes[0, 0].hist(top_df['sharpe_ratio'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Sharpe Ratio')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Sharpe Ratio Distribution')
axes[0, 0].axvline(2.0, color='red', linestyle='--', label='Sharpe = 2.0')
axes[0, 0].legend()

# Return vs Drawdown scatter
axes[0, 1].scatter(top_df['max_drawdown'], top_df['total_return'], 
                   s=top_df['total_trades'], alpha=0.6)
axes[0, 1].set_xlabel('Max Drawdown')
axes[0, 1].set_ylabel('Total Return')
axes[0, 1].set_title('Risk-Return Profile (size = # trades)')
axes[0, 1].grid(True, alpha=0.3)

# Win rate by strategy type
type_winrate = top_df.groupby('strategy_type')['win_rate'].mean().sort_values()
type_winrate.plot(kind='barh', ax=axes[1, 0])
axes[1, 0].axvline(0.5, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Win Rate')
axes[1, 0].set_title('Average Win Rate by Strategy Type')

# Trades vs Sharpe
axes[1, 1].scatter(top_df['total_trades'], top_df['sharpe_ratio'], alpha=0.6)
axes[1, 1].set_xlabel('Total Trades')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].set_title('Trading Frequency vs Performance')
axes[1, 1].set_xscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find strategies with best risk-adjusted returns
print("\nBest Risk-Adjusted Returns (Return/Drawdown Ratio):")
risk_adjusted = top_df.nlargest(5, 'return_drawdown_ratio')[
    ['strategy_type', 'sharpe_ratio', 'total_return', 'max_drawdown', 'return_drawdown_ratio']
]
print(risk_adjusted.to_string(index=False))

# Store for further analysis
top_performers = top_df
print(f"\nStored {len(top_performers)} strategies in 'top_performers' DataFrame")