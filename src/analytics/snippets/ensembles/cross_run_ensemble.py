# Build ensemble from multiple parameter sweep runs
# This snippet helps combine results from different indicator sweeps

# Parameters:
RUNS_TO_COMBINE = []  # Will be populated from notebook if available
MIN_SHARPE_PER_TYPE = 1.2
MAX_PER_TYPE = 3  # Maximum strategies per type for diversity
CORRELATION_CHECK = True
OUTPUT_PREFIX = "cross_run_ensemble"

# Check if we have combined_strategies from multi-run notebook
if 'combined_strategies' not in locals():
    print("⚠️ No combined strategies found. Loading manually...")
    
    # Manual loading if needed
    if not RUNS_TO_COMBINE:
        # Find recent runs
        import glob
        recent_runs = sorted(glob.glob("results/run_*"), reverse=True)[:5]
        print(f"Found {len(recent_runs)} recent runs")
        RUNS_TO_COMBINE = recent_runs
    
    # Load and combine
    all_strategies = []
    for run_dir in RUNS_TO_COMBINE:
        index_path = Path(run_dir) / 'strategy_index.parquet'
        if index_path.exists():
            strategies = pd.read_parquet(index_path)
            strategies['run_dir'] = run_dir
            all_strategies.append(strategies)
            print(f"Loaded {len(strategies)} strategies from {Path(run_dir).name}")
    
    if all_strategies:
        combined_strategies = pd.concat(all_strategies, ignore_index=True)
        # Remove duplicates by strategy hash
        combined_strategies = combined_strategies.drop_duplicates(subset=['strategy_hash'])
        print(f"\nTotal unique strategies: {len(combined_strategies)}")

# Build diverse ensemble
print(f"\nBuilding cross-run ensemble (min Sharpe: {MIN_SHARPE_PER_TYPE})...")

# Group by strategy type and select top performers
ensemble_candidates = []
type_summary = {}

for stype in combined_strategies['strategy_type'].unique():
    type_strategies = combined_strategies[
        (combined_strategies['strategy_type'] == stype) & 
        (combined_strategies['sharpe_ratio'] >= MIN_SHARPE_PER_TYPE)
    ]
    
    if len(type_strategies) > 0:
        # Take top N by Sharpe
        top_n = type_strategies.nlargest(MAX_PER_TYPE, 'sharpe_ratio')
        ensemble_candidates.append(top_n)
        
        type_summary[stype] = {
            'total': len(type_strategies),
            'selected': len(top_n),
            'best_sharpe': top_n['sharpe_ratio'].max(),
            'avg_sharpe': top_n['sharpe_ratio'].mean()
        }

# Combine candidates
if ensemble_candidates:
    ensemble_df = pd.concat(ensemble_candidates, ignore_index=True)
    print(f"\nEnsemble candidates: {len(ensemble_df)} strategies from {len(type_summary)} types")
    
    # Display type summary
    print("\nStrategy Type Summary:")
    print("-" * 60)
    for stype, stats in type_summary.items():
        print(f"{stype}:")
        print(f"  Qualified: {stats['total']} | Selected: {stats['selected']}")
        print(f"  Best Sharpe: {stats['best_sharpe']:.2f} | Avg: {stats['avg_sharpe']:.2f}")
    
    # Sort by Sharpe for final selection
    ensemble_df = ensemble_df.sort_values('sharpe_ratio', ascending=False)
    
    # Optional: Check correlations if we have signal data
    if CORRELATION_CHECK and len(ensemble_df) > 1:
        print("\n🔍 Checking correlations...")
        # This would require loading signals - for now just flag it
        print("💡 Run correlation analysis snippet to verify low correlations")
    
    # Display final ensemble
    print("\n🎯 Cross-Run Ensemble:")
    print("=" * 80)
    display_cols = ['strategy_type', 'strategy_hash', 'sharpe_ratio', 'total_return']
    if 'run_dir' in ensemble_df.columns:
        display_cols.append('run_dir')
    
    for idx, strategy in ensemble_df.iterrows():
        run_name = Path(strategy.get('run_dir', 'unknown')).name if 'run_dir' in strategy else 'unknown'
        print(f"\n{idx+1}. {strategy['strategy_type']} ({run_name})")
        print(f"   Hash: {strategy['strategy_hash'][:8]}")
        print(f"   Sharpe: {strategy['sharpe_ratio']:.2f} | Return: {strategy['total_return']:.1%}")
        
        # Show parameters if available
        param_cols = [col for col in strategy.index if col.startswith('param_')]
        if param_cols:
            params = {col.replace('param_', ''): strategy[col] for col in param_cols[:3] if pd.notna(strategy[col])}
            if params:
                print(f"   Params: {params}")
    
    # Calculate ensemble metrics
    print(f"\n📊 Ensemble Metrics:")
    print(f"  Total Strategies: {len(ensemble_df)}")
    print(f"  Average Sharpe: {ensemble_df['sharpe_ratio'].mean():.2f}")
    print(f"  Sharpe Range: {ensemble_df['sharpe_ratio'].min():.2f} - {ensemble_df['sharpe_ratio'].max():.2f}")
    print(f"  Strategy Types: {', '.join(ensemble_df['strategy_type'].unique())}")
    
    # Export configuration
    ensemble_config = {
        'created_at': pd.Timestamp.now().isoformat(),
        'selection_criteria': {
            'min_sharpe_per_type': MIN_SHARPE_PER_TYPE,
            'max_per_type': MAX_PER_TYPE,
            'runs_analyzed': len(set(ensemble_df.get('run_dir', []))) if 'run_dir' in ensemble_df.columns else 'unknown'
        },
        'strategies': [],
        'metrics': {
            'total_strategies': len(ensemble_df),
            'avg_sharpe': ensemble_df['sharpe_ratio'].mean(),
            'strategy_types': list(ensemble_df['strategy_type'].unique())
        }
    }
    
    for _, strategy in ensemble_df.iterrows():
        ensemble_config['strategies'].append({
            'strategy_hash': strategy['strategy_hash'],
            'strategy_type': strategy['strategy_type'],
            'sharpe_ratio': float(strategy['sharpe_ratio']),
            'total_return': float(strategy['total_return']),
            'source_run': Path(strategy.get('run_dir', 'unknown')).name if 'run_dir' in strategy else 'unknown',
            'weight': 1.0 / len(ensemble_df)  # Equal weight default
        })
    
    # Save configuration
    output_file = f"{OUTPUT_PREFIX}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f"\n✅ Ensemble configuration saved to: {output_file}")
    
    # Store for further analysis
    cross_run_ensemble = ensemble_df
else:
    print("❌ No strategies met the criteria")