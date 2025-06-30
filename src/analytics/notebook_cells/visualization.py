"""
Visualization cells for strategy analysis.
"""

def performance_heatmap_cell():
    """Cell for creating parameter performance heatmaps"""
    return """
# Create performance heatmaps for parameter optimization
def create_parameter_heatmap(performance_df, param1, param2, metric='sharpe_ratio'):
    '''Create heatmap showing performance across two parameters'''
    
    if param1 not in performance_df.columns or param2 not in performance_df.columns:
        print(f"Parameters {param1} or {param2} not found")
        return
    
    # Pivot data
    heatmap_data = performance_df.pivot_table(
        index=param1, 
        columns=param2, 
        values=metric,
        aggfunc='mean'  # Handle duplicates
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0 if metric == 'sharpe_ratio' else None,
                ax=ax)
    
    plt.title(f'{metric.replace("_", " ").title()} by {param1} and {param2}')
    plt.tight_layout()
    plt.show()
    
    # Find sweet spots
    if metric == 'sharpe_ratio':
        sweet_spots = performance_df[performance_df[metric] > sharpe_threshold]
        if len(sweet_spots) > 0:
            print(f"\\nParameter combinations with {metric} > {sharpe_threshold}:")
            print(sweet_spots[[param1, param2, metric]].sort_values(metric, ascending=False).head(10))

# Create heatmaps for different strategy types
for strategy_type in performance_df['strategy_type'].unique():
    type_data = performance_df[performance_df['strategy_type'] == strategy_type]
    
    print(f"\\n{strategy_type.upper()} Parameter Analysis")
    print("-" * 50)
    
    # Find numeric parameter columns
    param_cols = [col for col in type_data.columns 
                  if col.startswith('param_') and type_data[col].dtype in ['int64', 'float64']]
    
    if len(param_cols) >= 2:
        create_parameter_heatmap(type_data, param_cols[0], param_cols[1])
"""


def equity_curve_comparison_cell():
    """Cell for comparing multiple equity curves"""
    return """
# Compare equity curves of different strategies
def plot_equity_curves(strategies_list, market_data, normalize=True):
    '''Plot equity curves for multiple strategies'''
    
    plt.figure(figsize=(14, 8))
    
    # Plot buy & hold first
    if normalize:
        bh_curve = (1 + market_data['returns']).cumprod()
        plt.plot(market_data.index, bh_curve, label='Buy & Hold', 
                color='gray', linewidth=2, alpha=0.7)
    
    # Plot each strategy
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies_list)))
    
    for idx, strategy in enumerate(strategies_list):
        try:
            # Get equity curve
            equity = get_equity_curve(strategy['strategy_hash'], market_data)
            
            if normalize:
                # Normalize to start at 1
                equity = equity / equity.iloc[0]
            
            label = f"{strategy['strategy_type']} ({strategy['strategy_hash'][:6]})"
            plt.plot(equity.index, equity.values, label=label, 
                    color=colors[idx], linewidth=1.5, alpha=0.8)
            
        except Exception as e:
            print(f"Failed to plot {strategy['strategy_hash']}: {e}")
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns' if normalize else 'Portfolio Value')
    plt.title('Strategy Performance Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add drawdown shading
    if len(strategies_list) > 0:
        # Use best strategy for drawdown shading
        best_equity = get_equity_curve(strategies_list[0]['strategy_hash'], market_data)
        if normalize:
            best_equity = best_equity / best_equity.iloc[0]
        
        cummax = best_equity.expanding().max()
        drawdown = (best_equity / cummax - 1)
        
        # Shade drawdown periods
        plt.fill_between(drawdown.index, 0, drawdown.values, 
                        where=(drawdown < 0), color='red', alpha=0.1)
    
    plt.tight_layout()
    plt.show()

# Plot top strategies
if 'top_performers' in locals() and len(top_performers) > 0:
    print("\\nComparing Top Strategy Equity Curves")
    plot_equity_curves(top_performers.head(5).to_dict('records'), market_data)
"""


def signal_distribution_cell():
    """Cell for visualizing signal distributions"""
    return """
# Analyze signal distribution patterns
def plot_signal_distribution(strategy_data, strategy_info):
    '''Visualize when and how signals are generated'''
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Signals by hour of day
    hourly = strategy_data.groupby(strategy_data.index.hour)['signal'].agg(['sum', 'count'])
    axes[0, 0].bar(hourly.index, hourly['count'])
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Signal Count')
    axes[0, 0].set_title('Signals by Hour')
    
    # 2. Signals by day of week
    daily = strategy_data.groupby(strategy_data.index.dayofweek)['signal'].agg(['sum', 'count'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    axes[0, 1].bar(range(5), daily['count'][:5])
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].set_xticklabels(days)
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Signal Count')
    axes[0, 1].set_title('Signals by Day of Week')
    
    # 3. Signal duration distribution
    signal_changes = strategy_data['signal'].diff() != 0
    signal_durations = signal_changes.cumsum().value_counts().sort_index()
    axes[1, 0].hist(signal_durations.values, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Duration (bars)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Signal Duration Distribution')
    axes[1, 0].set_yscale('log')
    
    # 4. Signal strength over time (if available)
    if 'signal_strength' in strategy_data.columns:
        strategy_data['signal_strength'].rolling(100).mean().plot(ax=axes[1, 1])
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Signal Strength (100-bar MA)')
        axes[1, 1].set_title('Signal Strength Over Time')
    else:
        # Cumulative signals
        strategy_data['signal'].abs().cumsum().plot(ax=axes[1, 1])
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Signals')
        axes[1, 1].set_title('Cumulative Signal Count')
    
    plt.suptitle(f"Signal Distribution Analysis: {strategy_info['strategy_type']} - {strategy_info['strategy_hash'][:8]}")
    plt.tight_layout()
    plt.show()

# Analyze signal distribution for best strategy
if len(performance_df) > 0:
    best = performance_df.iloc[0]
    strategy_data = get_strategy_data(best['strategy_hash'])
    plot_signal_distribution(strategy_data, best)
"""