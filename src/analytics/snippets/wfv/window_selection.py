# Walk-Forward Validation: Window Parameter Selection
# This snippet helps select robust parameters for the next OOS period

# Parameters for selection criteria
MIN_IS_SHARPE = 1.2          # In-sample minimum Sharpe
MIN_TRADES = 50              # Minimum trades for statistical significance
STABILITY_RADIUS = 2         # Check neighboring parameters ±N steps
MAX_PARAMETERS = 5           # Maximum parameters to forward test
REGIME_DIVERSITY = True      # Require performance in multiple regimes

print(f"WFV Window Parameter Selection")
print(f"Current Window: {wfv_window if 'wfv_window' in locals() else 'Unknown'}")
print("=" * 60)

# Check if we have IS results
if 'performance_df' not in locals():
    print("⚠️ No performance_df found. Run the analysis notebook first.")
else:
    # Filter by minimum Sharpe
    candidates = performance_df[performance_df['sharpe_ratio'] >= MIN_IS_SHARPE].copy()
    print(f"\nCandidates with Sharpe >= {MIN_IS_SHARPE}: {len(candidates)}")
    
    # Add stability score
    if STABILITY_RADIUS > 0:
        print(f"\nChecking parameter stability (radius={STABILITY_RADIUS})...")
        
        # For each candidate, check neighboring parameters
        stability_scores = []
        
        for idx, candidate in candidates.iterrows():
            stability_score = 0
            neighbors_checked = 0
            
            # Get parameter columns
            param_cols = [col for col in candidate.index if col.startswith('param_')]
            
            # Check neighboring parameters
            for param_col in param_cols:
                if pd.notna(candidate[param_col]) and candidate[param_col] != 0:
                    param_val = candidate[param_col]
                    
                    # Look for neighbors
                    for delta in range(-STABILITY_RADIUS, STABILITY_RADIUS + 1):
                        if delta == 0:
                            continue
                        
                        neighbor_val = param_val + delta
                        neighbor_mask = abs(performance_df[param_col] - neighbor_val) < 0.01
                        
                        if neighbor_mask.any():
                            neighbor_sharpe = performance_df.loc[neighbor_mask, 'sharpe_ratio'].mean()
                            if neighbor_sharpe > MIN_IS_SHARPE * 0.8:  # 80% of threshold
                                stability_score += 1
                            neighbors_checked += 1
            
            stability_scores.append(
                stability_score / neighbors_checked if neighbors_checked > 0 else 0
            )
        
        candidates['stability_score'] = stability_scores
        candidates = candidates[candidates['stability_score'] > 0.5]  # At least 50% stable neighbors
        print(f"Stable candidates: {len(candidates)}")
    
    # Check regime diversity if requested
    if REGIME_DIVERSITY and 'regime_performance' in locals():
        print("\nChecking regime diversity...")
        diverse_candidates = []
        
        for idx, candidate in candidates.iterrows():
            strategy_hash = candidate['strategy_hash']
            
            # Check how many regimes this strategy performs in
            strategy_regimes = regime_performance[
                regime_performance['strategy_hash'] == strategy_hash
            ]
            
            if len(strategy_regimes['volatility_regime'].unique()) >= 2:
                diverse_candidates.append(idx)
        
        candidates = candidates.loc[diverse_candidates]
        print(f"Regime-diverse candidates: {len(candidates)}")
    
    # Rank candidates by combined score
    if len(candidates) > 0:
        # Calculate combined score
        candidates['combined_score'] = (
            candidates['sharpe_ratio'] * 0.5 +
            candidates.get('stability_score', 1.0) * 0.3 +
            (1 / (candidates['max_drawdown'].abs() + 0.01)) * 0.2
        )
        
        # Select top N
        selected = candidates.nlargest(MAX_PARAMETERS, 'combined_score')
        
        print(f"\n🎯 Selected {len(selected)} parameters for OOS testing:")
        print("-" * 60)
        
        for i, (idx, strategy) in enumerate(selected.iterrows(), 1):
            print(f"\n{i}. {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
            print(f"   IS Sharpe: {strategy['sharpe_ratio']:.2f}")
            print(f"   Stability: {strategy.get('stability_score', 'N/A'):.2f}")
            print(f"   Combined Score: {strategy['combined_score']:.2f}")
            
            # Show parameters
            param_cols = [col for col in strategy.index if col.startswith('param_')]
            params = {col.replace('param_', ''): strategy[col] 
                     for col in param_cols if pd.notna(strategy[col])}
            if params:
                print(f"   Parameters: {params}")
        
        # Export selections
        wfv_selections = {
            'window': wfv_window if 'wfv_window' in locals() else 'unknown',
            'selection_criteria': {
                'min_is_sharpe': MIN_IS_SHARPE,
                'min_trades': MIN_TRADES,
                'stability_radius': STABILITY_RADIUS,
                'regime_diversity': REGIME_DIVERSITY
            },
            'selected_strategies': selected.to_dict('records'),
            'selection_timestamp': pd.Timestamp.now().isoformat()
        }
        
        output_file = f"wfv_window_{wfv_window if 'wfv_window' in locals() else 'X'}_selections.json"
        with open(output_file, 'w') as f:
            json.dump(wfv_selections, f, indent=2, default=str)
        
        print(f"\n✅ Selections saved to {output_file}")
        
        # Store for further analysis
        wfv_selected_parameters = selected
    else:
        print("\n❌ No candidates met all criteria")
        
        # Diagnostic info
        print("\nDiagnostic Information:")
        print(f"  Total strategies: {len(performance_df)}")
        print(f"  With min Sharpe: {len(performance_df[performance_df['sharpe_ratio'] >= MIN_IS_SHARPE])}")
        print(f"  Max Sharpe found: {performance_df['sharpe_ratio'].max():.2f}")
        
        # Suggest relaxing criteria
        print("\nSuggestions:")
        print("  1. Lower MIN_IS_SHARPE")
        print("  2. Reduce STABILITY_RADIUS")
        print("  3. Disable REGIME_DIVERSITY")