# Portfolio weight optimization using mean-variance optimization
# Edit parameters:
RISK_FREE_RATE = 0.02 / 252    # Daily risk-free rate
MAX_WEIGHT = 0.4               # Maximum weight per strategy
MIN_WEIGHT = 0.0               # Minimum weight (0 = allow no allocation)
TARGET_VOL = None              # Target volatility (None = maximize Sharpe)
REBALANCE_THRESHOLD = 0.05     # Minimum weight change to trigger rebalance

# Check if we have an ensemble to optimize
if 'uncorrelated_ensemble' not in locals():
    print("⚠️ No ensemble found. Run find_uncorrelated.py first or define uncorrelated_ensemble")
else:
    print(f"Optimizing weights for {len(uncorrelated_ensemble)} strategies...")
    
    # Get returns for each strategy
    returns_data = {}
    
    for idx, strategy in uncorrelated_ensemble.iterrows():
        try:
            # Load signals
            signals = pd.read_parquet(run_dir / strategy['trace_path'])
            signals['ts'] = pd.to_datetime(signals['ts'])
            
            # Calculate returns
            df = market_data.merge(signals[['ts', 'val']], 
                                  left_on='timestamp', right_on='ts', how='left')
            df['signal'] = df['val'].fillna(method='ffill').fillna(0)
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
            
            returns_data[strategy['strategy_hash']] = df['strategy_returns']
            
        except Exception as e:
            print(f"Failed to load returns for {strategy['strategy_hash']}: {e}")
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_data).dropna()
    print(f"Using {len(returns_df)} days of return data")
    
    # Calculate expected returns and covariance
    expected_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    # Mean-Variance Optimization
    from scipy.optimize import minimize
    n_assets = len(expected_returns)
    
    def portfolio_stats(weights):
        """Calculate portfolio return and volatility"""
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return portfolio_return, portfolio_vol
    
    def neg_sharpe(weights):
        """Negative Sharpe ratio for minimization"""
        p_ret, p_vol = portfolio_stats(weights)
        return -(p_ret - RISK_FREE_RATE) / p_vol
    
    def portfolio_vol(weights):
        """Portfolio volatility"""
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    if TARGET_VOL is not None:
        # Add volatility constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_vol(x) - TARGET_VOL
        })
    
    # Bounds
    bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n_assets))
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    print("\nOptimizing portfolio weights...")
    result = minimize(neg_sharpe, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'disp': True})
    
    if result.success:
        optimal_weights = result.x
        opt_return, opt_vol = portfolio_stats(optimal_weights)
        opt_sharpe = (opt_return - RISK_FREE_RATE) / opt_vol
        
        print("\n✅ Optimization successful!")
        print(f"Optimal Sharpe: {opt_sharpe * np.sqrt(252 * 78):.2f} (annualized)")
        print(f"Expected Return: {opt_return * 252 * 78:.1%} (annualized)")
        print(f"Volatility: {opt_vol * np.sqrt(252 * 78):.1%} (annualized)")
        
        # Display weights
        print("\nOptimal Weights:")
        weight_df = pd.DataFrame({
            'strategy_hash': list(returns_data.keys()),
            'strategy_type': uncorrelated_ensemble['strategy_type'].values,
            'individual_sharpe': uncorrelated_ensemble['sharpe_ratio'].values,
            'weight': optimal_weights,
            'weight_pct': optimal_weights * 100
        }).sort_values('weight', ascending=False)
        
        # Only show non-zero weights
        weight_df_nonzero = weight_df[weight_df['weight'] > 0.001]
        print(weight_df_nonzero.to_string(index=False))
        
        # Compare with equal weight
        equal_weights = np.ones(n_assets) / n_assets
        eq_return, eq_vol = portfolio_stats(equal_weights)
        eq_sharpe = (eq_return - RISK_FREE_RATE) / eq_vol
        
        print(f"\nEqual Weight Sharpe: {eq_sharpe * np.sqrt(252 * 78):.2f}")
        print(f"Improvement: {((opt_sharpe/eq_sharpe - 1) * 100):.1f}%")
        
        # Visualize weights
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of weights
        nonzero_weights = weight_df_nonzero[weight_df_nonzero['weight'] > REBALANCE_THRESHOLD]
        if len(nonzero_weights) > 0:
            ax1.pie(nonzero_weights['weight'], 
                   labels=[f"{row['strategy_type'][:4]}..{row['strategy_hash'][-4:]}" 
                          for _, row in nonzero_weights.iterrows()],
                   autopct='%1.1f%%')
            ax1.set_title('Portfolio Allocation')
        
        # Efficient frontier
        # Generate random portfolios
        n_portfolios = 5000
        results = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            p_ret, p_vol = portfolio_stats(weights)
            results[0, i] = p_ret * 252 * 78  # Annualized
            results[1, i] = p_vol * np.sqrt(252 * 78)  # Annualized
            results[2, i] = (p_ret - RISK_FREE_RATE) / p_vol * np.sqrt(252 * 78)  # Annualized Sharpe
        
        # Plot
        scatter = ax2.scatter(results[1], results[0], c=results[2], 
                            cmap='viridis', alpha=0.5, s=10)
        ax2.scatter(opt_vol * np.sqrt(252 * 78), opt_return * 252 * 78, 
                   marker='*', s=500, c='red', label='Optimal Portfolio')
        ax2.scatter(eq_vol * np.sqrt(252 * 78), eq_return * 252 * 78, 
                   marker='o', s=200, c='orange', label='Equal Weight')
        
        ax2.set_xlabel('Volatility (Annualized)')
        ax2.set_ylabel('Return (Annualized)')
        ax2.set_title('Efficient Frontier')
        ax2.legend()
        
        plt.colorbar(scatter, ax=ax2, label='Sharpe Ratio')
        plt.tight_layout()
        plt.show()
        
        # Store results
        optimal_portfolio = {
            'weights': dict(zip(returns_data.keys(), optimal_weights)),
            'expected_return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': opt_sharpe,
            'weight_df': weight_df_nonzero
        }
        
        print("\nResults stored in 'optimal_portfolio' dictionary")
        
    else:
        print("❌ Optimization failed:", result.message)