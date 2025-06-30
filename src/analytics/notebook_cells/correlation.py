"""
Correlation analysis cells for ensemble building.
"""

def correlation_matrix_cell():
    """Cell for creating correlation matrices"""
    return """
# Calculate correlation matrix for top strategies
def calculate_correlation_matrix(strategies_list, run_dir, min_overlap=100):
    '''Calculate pairwise correlations between strategy signals'''
    
    # Load all signals
    signals_dict = {}
    for strategy in strategies_list:
        try:
            trace_path = run_dir / strategy['trace_path']
            signals = pd.read_parquet(trace_path)
            signals['ts'] = pd.to_datetime(signals['ts'])
            signals = signals.set_index('ts')['val']
            signals_dict[strategy['strategy_hash']] = signals
        except Exception as e:
            print(f"Failed to load {strategy['strategy_hash']}: {e}")
    
    # Create aligned DataFrame
    signals_df = pd.DataFrame(signals_dict)
    
    # Forward fill to create dense signals
    signals_df = signals_df.fillna(method='ffill').fillna(0)
    
    # Calculate correlations only where sufficient overlap
    corr_matrix = signals_df.corr(min_periods=min_overlap)
    
    return corr_matrix, signals_df

# Calculate correlations
if 'top_performers' in locals() and len(top_performers) > 1:
    print("Calculating correlation matrix...")
    corr_matrix, aligned_signals = calculate_correlation_matrix(
        top_performers.to_dict('records'), 
        run_dir
    )
    
    # Visualize
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
                annot=True, fmt='.2f', vmin=-1, vmax=1)
    plt.title('Strategy Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find uncorrelated pairs
    uncorrelated = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) < correlation_threshold:
                uncorrelated.append({
                    'strategy1': corr_matrix.index[i],
                    'strategy2': corr_matrix.columns[j],
                    'correlation': corr
                })
    
    print(f"\\nFound {len(uncorrelated)} uncorrelated pairs (|corr| < {correlation_threshold})")
"""


def ensemble_optimization_cell():
    """Cell for optimizing ensemble weights"""
    return """
# Optimize ensemble weights using mean-variance optimization
def optimize_ensemble_weights(returns_df, target_return=None):
    '''Find optimal portfolio weights using Markowitz optimization'''
    
    from scipy.optimize import minimize
    
    # Calculate expected returns and covariance
    expected_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    n_assets = len(returns_df.columns)
    
    # Objective: minimize portfolio variance
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    # Constraint: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Bounds: 0 <= weight <= 1 for each asset
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(portfolio_variance, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = expected_returns @ optimal_weights
        portfolio_std = np.sqrt(portfolio_variance(optimal_weights))
        portfolio_sharpe = portfolio_return / portfolio_std * np.sqrt(252 * 78)
        
        return {
            'weights': dict(zip(returns_df.columns, optimal_weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': portfolio_sharpe
        }
    else:
        return None

# Optimize ensemble if we have uncorrelated strategies
if 'uncorrelated' in locals() and len(uncorrelated) >= 2:
    # Get returns for uncorrelated strategies
    ensemble_candidates = set()
    for pair in uncorrelated[:10]:  # Limit to avoid too many
        ensemble_candidates.add(pair['strategy1'])
        ensemble_candidates.add(pair['strategy2'])
    
    ensemble_candidates = list(ensemble_candidates)[:ensemble_size]
    
    print(f"\\nOptimizing ensemble of {len(ensemble_candidates)} strategies...")
    
    # Get returns data
    ensemble_returns = pd.DataFrame()
    for strategy_hash in ensemble_candidates:
        returns = get_strategy_returns(strategy_hash)
        ensemble_returns[strategy_hash[:8]] = returns
    
    # Optimize
    optimal = optimize_ensemble_weights(ensemble_returns)
    
    if optimal:
        print("\\nOptimal Ensemble Weights:")
        for strategy, weight in optimal['weights'].items():
            if weight > 0.01:  # Show only significant weights
                print(f"  {strategy}: {weight:.1%}")
        
        print(f"\\nEnsemble Metrics:")
        print(f"  Expected Return: {optimal['expected_return']:.2%} daily")
        print(f"  Volatility: {optimal['volatility']:.2%} daily")
        print(f"  Sharpe Ratio: {optimal['sharpe_ratio']:.2f}")
"""


def rolling_correlation_cell():
    """Cell for analyzing time-varying correlations"""
    return """
# Analyze rolling correlations to detect regime changes
def calculate_rolling_correlations(signals_df, window=252):
    '''Calculate rolling correlations between strategies'''
    
    # Pick two strategies to analyze
    if signals_df.shape[1] >= 2:
        strat1, strat2 = signals_df.columns[:2]
        
        # Calculate rolling correlation
        rolling_corr = signals_df[strat1].rolling(window).corr(signals_df[strat2])
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Signals
        ax1.plot(signals_df.index, signals_df[strat1], label=strat1[:8], alpha=0.7)
        ax1.plot(signals_df.index, signals_df[strat2], label=strat2[:8], alpha=0.7)
        ax1.set_ylabel('Signal')
        ax1.legend()
        ax1.set_title('Strategy Signals')
        
        # Rolling correlation
        ax2.plot(rolling_corr.index, rolling_corr.values)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(y=correlation_threshold, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=-correlation_threshold, color='r', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Correlation')
        ax2.set_xlabel('Date')
        ax2.set_title(f'{window}-Bar Rolling Correlation')
        
        plt.tight_layout()
        plt.show()
        
        # Identify correlation regimes
        high_corr_periods = (rolling_corr > correlation_threshold).sum()
        low_corr_periods = (abs(rolling_corr) < 0.3).sum()
        
        print(f"\\nCorrelation Regime Analysis:")
        print(f"  High correlation periods (>{correlation_threshold}): {high_corr_periods/len(rolling_corr):.1%}")
        print(f"  Low correlation periods (<0.3): {low_corr_periods/len(rolling_corr):.1%}")
        print(f"  Average correlation: {rolling_corr.mean():.3f}")
        print(f"  Correlation volatility: {rolling_corr.std():.3f}")

# Run rolling correlation analysis
if 'aligned_signals' in locals() and aligned_signals.shape[1] >= 2:
    calculate_rolling_correlations(aligned_signals)
"""