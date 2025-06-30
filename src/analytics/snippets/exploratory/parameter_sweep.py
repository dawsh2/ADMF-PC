# Parameter sensitivity analysis - find optimal parameter ranges
# Edit the strategy type and parameters to analyze:
STRATEGY_TYPE = 'momentum'  # Change to your strategy type
PARAM1 = 'param_fast_period'  # First parameter to analyze
PARAM2 = 'param_slow_period'  # Second parameter to analyze
MIN_INSTANCES = 3  # Minimum instances per parameter combination

# Get all strategies of this type
param_sql = f"""
SELECT 
    strategy_hash,
    strategy_type,
    sharpe_ratio,
    total_return,
    max_drawdown,
    win_rate,
    total_trades,
    param_names,
    param_values
FROM strategies
WHERE strategy_type = '{STRATEGY_TYPE}'
    AND sharpe_ratio IS NOT NULL
"""

print(f"Analyzing {STRATEGY_TYPE} parameter sensitivity...")
param_df = con.execute(param_sql).df()
print(f"Found {len(param_df)} {STRATEGY_TYPE} strategies")

# Extract parameter columns
param_cols = [col for col in param_df.columns if col.startswith('param_')]
print(f"Available parameters: {param_cols}")

# If parameters are in JSON format, extract them
if 'param_values' in param_df.columns and param_df['param_values'].notna().any():
    try:
        import json
        # Extract parameters from JSON
        params_expanded = param_df['param_values'].apply(
            lambda x: json.loads(x) if pd.notna(x) else {}
        )
        for key in params_expanded.iloc[0].keys():
            param_df[f'param_{key}'] = params_expanded.apply(lambda x: x.get(key))
        param_cols = [col for col in param_df.columns if col.startswith('param_')]
    except:
        pass

# Single parameter analysis
if PARAM1 in param_cols:
    param1_analysis = param_df.groupby(PARAM1).agg({
        'sharpe_ratio': ['count', 'mean', 'std', 'max'],
        'total_return': 'mean',
        'max_drawdown': 'mean',
        'win_rate': 'mean'
    }).round(3)
    
    # Filter for statistical significance
    param1_analysis = param1_analysis[param1_analysis[('sharpe_ratio', 'count')] >= MIN_INSTANCES]
    
    print(f"\n{PARAM1} Analysis:")
    print(param1_analysis)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    param1_values = param1_analysis.index
    param1_sharpe = param1_analysis[('sharpe_ratio', 'mean')]
    param1_std = param1_analysis[('sharpe_ratio', 'std')]
    
    ax.errorbar(param1_values, param1_sharpe, yerr=param1_std, 
                marker='o', capsize=5, capthick=2)
    ax.set_xlabel(PARAM1)
    ax.set_ylabel('Average Sharpe Ratio')
    ax.set_title(f'{STRATEGY_TYPE}: Sharpe Ratio vs {PARAM1}')
    ax.grid(True, alpha=0.3)
    plt.show()

# Two-parameter heatmap analysis
if PARAM1 in param_cols and PARAM2 in param_cols:
    # Create pivot table
    param_pivot = param_df.pivot_table(
        index=PARAM1,
        columns=PARAM2,
        values='sharpe_ratio',
        aggfunc='mean'
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = param_pivot.isna()
    sns.heatmap(param_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, mask=mask, cbar_kws={'label': 'Average Sharpe Ratio'})
    plt.title(f'{STRATEGY_TYPE}: Parameter Interaction Heatmap')
    plt.xlabel(PARAM2)
    plt.ylabel(PARAM1)
    plt.tight_layout()
    plt.show()
    
    # Find sweet spots
    sweet_spots = param_df[
        (param_df['sharpe_ratio'] > param_df['sharpe_ratio'].quantile(0.8))
    ].groupby([PARAM1, PARAM2]).agg({
        'sharpe_ratio': ['count', 'mean'],
        'total_return': 'mean'
    })
    
    print(f"\nParameter Sweet Spots (top 20% Sharpe):")
    print(sweet_spots.sort_values(('sharpe_ratio', 'mean'), ascending=False).head(10))

# Parameter importance analysis
print("\nParameter Correlation with Performance:")
numeric_params = param_df.select_dtypes(include=[np.number]).columns
param_only = [col for col in numeric_params if col.startswith('param_')]
if param_only:
    correlations = param_df[param_only + ['sharpe_ratio']].corr()['sharpe_ratio'][param_only]
    correlations = correlations.sort_values(ascending=False)
    print(correlations)
    
    # Visualize correlations
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='barh')
    plt.xlabel('Correlation with Sharpe Ratio')
    plt.title(f'{STRATEGY_TYPE}: Parameter Importance')
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Store results
parameter_analysis = param_df
print(f"\nStored {len(parameter_analysis)} strategies in 'parameter_analysis' DataFrame")