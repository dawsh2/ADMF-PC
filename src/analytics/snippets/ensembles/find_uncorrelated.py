# Find uncorrelated strategies for ensemble construction
# Edit these parameters:
MIN_SHARPE = 1.5              # Minimum Sharpe for candidates
MIN_TRADES = 20               # Minimum trades for statistical significance  
CORRELATION_THRESHOLD = 0.5   # Maximum acceptable correlation
MIN_OBSERVATIONS = 100        # Minimum overlapping periods
TOP_N_CANDIDATES = 30         # Number of candidates to consider

# Load correlation query
with open('src/analytics/queries/correlation_pairs.sql', 'r') as f:
    correlation_sql = f.read()

# First, get top candidates
print(f"Finding top {TOP_N_CANDIDATES} candidates with Sharpe > {MIN_SHARPE}...")
candidates_df = con.execute(f"""
    SELECT 
        strategy_hash,
        strategy_type,
        sharpe_ratio,
        total_return,
        max_drawdown,
        total_trades
    FROM strategies
    WHERE sharpe_ratio > {MIN_SHARPE}
        AND total_trades >= {MIN_TRADES}
    ORDER BY sharpe_ratio DESC
    LIMIT {TOP_N_CANDIDATES}
""").df()

print(f"Found {len(candidates_df)} candidate strategies")
print("\nTop 10 candidates:")
print(candidates_df.head(10))

# Find uncorrelated pairs
print(f"\nFinding pairs with correlation < {CORRELATION_THRESHOLD}...")
uncorrelated_pairs = con.execute(correlation_sql.format(
    min_observations=MIN_OBSERVATIONS,
    correlation_threshold=CORRELATION_THRESHOLD,
    min_sharpe=MIN_SHARPE
)).df()

print(f"Found {len(uncorrelated_pairs)} uncorrelated pairs")

# Build correlation matrix for visualization
if len(candidates_df) > 1:
    # Get signals for correlation calculation
    print("\nCalculating full correlation matrix...")
    
    # Create a view of candidate signals
    candidate_hashes = "','".join(candidates_df['strategy_hash'].tolist())
    signals_matrix_sql = f"""
    WITH signal_data AS (
        SELECT 
            strategy_hash,
            DATE_TRUNC('hour', ts) as hour_ts,
            AVG(val) as avg_signal
        FROM signals
        WHERE strategy_hash IN ('{candidate_hashes}')
            AND val != 0
        GROUP BY strategy_hash, hour_ts
    )
    SELECT 
        hour_ts,
        strategy_hash,
        avg_signal
    FROM signal_data
    """
    
    signals_df = con.execute(signals_matrix_sql).df()
    
    # Pivot to create matrix
    if not signals_df.empty:
        correlation_matrix = signals_df.pivot(
            index='hour_ts', 
            columns='strategy_hash', 
            values='avg_signal'
        ).corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create labels with strategy types
        labels = []
        for hash in correlation_matrix.index:
            strategy_type = candidates_df[
                candidates_df['strategy_hash'] == hash
            ]['strategy_type'].iloc[0] if hash in candidates_df['strategy_hash'].values else 'unknown'
            labels.append(f"{strategy_type[:4]}..{hash[-4:]}")
        
        sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1, annot=len(correlation_matrix) < 15,
                    fmt='.2f', xticklabels=labels, yticklabels=labels)
        plt.title('Strategy Correlation Matrix')
        plt.tight_layout()
        plt.show()

# Find optimal uncorrelated set
print("\nBuilding optimal uncorrelated ensemble...")
selected_strategies = []
remaining_candidates = candidates_df['strategy_hash'].tolist()

# Greedy selection algorithm
while len(selected_strategies) < 10 and remaining_candidates:
    if not selected_strategies:
        # Start with highest Sharpe
        best = remaining_candidates[0]
        selected_strategies.append(best)
        remaining_candidates.remove(best)
    else:
        # Find least correlated candidate
        best_candidate = None
        min_max_correlation = 1.0
        
        for candidate in remaining_candidates:
            # Check correlation with all selected strategies
            max_corr = 0
            for selected in selected_strategies:
                # Find correlation between candidate and selected
                pair_corr = uncorrelated_pairs[
                    ((uncorrelated_pairs['strategy1'] == candidate) & 
                     (uncorrelated_pairs['strategy2'] == selected)) |
                    ((uncorrelated_pairs['strategy2'] == candidate) & 
                     (uncorrelated_pairs['strategy1'] == selected))
                ]['signal_correlation']
                
                if not pair_corr.empty:
                    max_corr = max(max_corr, abs(pair_corr.iloc[0]))
                else:
                    max_corr = 1.0  # Assume high correlation if no data
            
            if max_corr < min_max_correlation:
                min_max_correlation = max_corr
                best_candidate = candidate
        
        if best_candidate and min_max_correlation < CORRELATION_THRESHOLD:
            selected_strategies.append(best_candidate)
            remaining_candidates.remove(best_candidate)
        else:
            break

# Display selected ensemble
ensemble_df = candidates_df[candidates_df['strategy_hash'].isin(selected_strategies)]
print(f"\nSelected {len(ensemble_df)} strategies for ensemble:")
print(ensemble_df[['strategy_type', 'sharpe_ratio', 'total_return', 'max_drawdown']])

# Calculate ensemble metrics
ensemble_metrics = {
    'num_strategies': len(ensemble_df),
    'avg_sharpe': ensemble_df['sharpe_ratio'].mean(),
    'min_sharpe': ensemble_df['sharpe_ratio'].min(),
    'avg_return': ensemble_df['total_return'].mean(),
    'strategies': ensemble_df.to_dict('records')
}

print(f"\nEnsemble Summary:")
print(f"  Strategies: {ensemble_metrics['num_strategies']}")
print(f"  Average Sharpe: {ensemble_metrics['avg_sharpe']:.2f}")
print(f"  Average Return: {ensemble_metrics['avg_return']:.1%}")

# Store results
uncorrelated_ensemble = ensemble_df
ensemble_pairs = uncorrelated_pairs