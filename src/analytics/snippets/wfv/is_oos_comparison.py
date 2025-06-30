# Walk-Forward Validation: IS/OOS Performance Comparison
# Compare in-sample selection performance with out-of-sample results

# Parameters
ACCEPTABLE_DEGRADATION = 0.3  # 30% degradation is acceptable
MIN_OOS_SHARPE = 0.8         # Minimum acceptable OOS Sharpe
PLOT_COMPARISON = True       # Create visualization

print("IS/OOS Performance Comparison")
print("=" * 60)

# Load IS selections if available
import glob
selection_files = sorted(glob.glob("wfv_window_*_selections.json"))

if selection_files:
    print(f"Found {len(selection_files)} window selection files")
    
    # Load most recent or all
    comparisons = []
    
    for selection_file in selection_files:
        with open(selection_file, 'r') as f:
            selections = json.load(f)
        
        window = selections['window']
        print(f"\nWindow {window}:")
        print("-" * 40)
        
        # Get IS performance from selections
        is_strategies = pd.DataFrame(selections['selected_strategies'])
        
        # If we have OOS results loaded
        if 'performance_df' in locals() and 'dataset' in locals() and dataset == 'test':
            # Match strategies by hash
            for _, is_strategy in is_strategies.iterrows():
                strategy_hash = is_strategy['strategy_hash']
                
                # Find OOS performance
                oos_match = performance_df[performance_df['strategy_hash'] == strategy_hash]
                
                if not oos_match.empty:
                    oos_strategy = oos_match.iloc[0]
                    
                    comparison = {
                        'window': window,
                        'strategy_hash': strategy_hash,
                        'strategy_type': is_strategy['strategy_type'],
                        'is_sharpe': is_strategy['sharpe_ratio'],
                        'oos_sharpe': oos_strategy['sharpe_ratio'],
                        'is_return': is_strategy['total_return'],
                        'oos_return': oos_strategy['total_return'],
                        'is_drawdown': is_strategy['max_drawdown'],
                        'oos_drawdown': oos_strategy['max_drawdown'],
                        'degradation': (oos_strategy['sharpe_ratio'] - is_strategy['sharpe_ratio']) / is_strategy['sharpe_ratio']
                    }
                    
                    comparisons.append(comparison)
                    
                    # Print summary
                    print(f"\n{is_strategy['strategy_type']} - {strategy_hash[:8]}")
                    print(f"  IS Sharpe:  {is_strategy['sharpe_ratio']:.2f}")
                    print(f"  OOS Sharpe: {oos_strategy['sharpe_ratio']:.2f}")
                    print(f"  Degradation: {comparison['degradation']:.1%}")
                    
                    status = "✅ PASS" if (
                        comparison['degradation'] > -ACCEPTABLE_DEGRADATION and 
                        oos_strategy['sharpe_ratio'] > MIN_OOS_SHARPE
                    ) else "❌ FAIL"
                    print(f"  Status: {status}")
    
    if comparisons:
        comparison_df = pd.DataFrame(comparisons)
        
        # Summary statistics
        print("\n\nOverall Statistics:")
        print("=" * 60)
        print(f"Total IS/OOS pairs: {len(comparison_df)}")
        print(f"Average IS Sharpe: {comparison_df['is_sharpe'].mean():.2f}")
        print(f"Average OOS Sharpe: {comparison_df['oos_sharpe'].mean():.2f}")
        print(f"Average Degradation: {comparison_df['degradation'].mean():.1%}")
        print(f"Strategies maintaining profitability: {(comparison_df['oos_sharpe'] > 0).sum()} ({(comparison_df['oos_sharpe'] > 0).mean():.1%})")
        
        # Visualization
        if PLOT_COMPARISON and len(comparison_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # IS vs OOS Sharpe
            axes[0, 0].scatter(comparison_df['is_sharpe'], comparison_df['oos_sharpe'])
            axes[0, 0].plot([0, 3], [0, 3], 'k--', alpha=0.5, label='No degradation')
            axes[0, 0].plot([0, 3], [0, 2.1], 'r--', alpha=0.5, label='30% degradation')
            axes[0, 0].set_xlabel('IS Sharpe Ratio')
            axes[0, 0].set_ylabel('OOS Sharpe Ratio')
            axes[0, 0].set_title('IS vs OOS Sharpe Ratio')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Degradation distribution
            axes[0, 1].hist(comparison_df['degradation'], bins=20, edgecolor='black')
            axes[0, 1].axvline(-ACCEPTABLE_DEGRADATION, color='r', linestyle='--', 
                              label=f'Acceptable ({-ACCEPTABLE_DEGRADATION:.0%})')
            axes[0, 1].set_xlabel('Degradation Rate')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('OOS Degradation Distribution')
            axes[0, 1].legend()
            
            # Returns comparison
            axes[1, 0].scatter(comparison_df['is_return'], comparison_df['oos_return'])
            axes[1, 0].plot([comparison_df['is_return'].min(), comparison_df['is_return'].max()],
                           [comparison_df['is_return'].min(), comparison_df['is_return'].max()],
                           'k--', alpha=0.5)
            axes[1, 0].set_xlabel('IS Total Return')
            axes[1, 0].set_ylabel('OOS Total Return')
            axes[1, 0].set_title('IS vs OOS Returns')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Degradation by strategy type
            degradation_by_type = comparison_df.groupby('strategy_type')['degradation'].mean()
            degradation_by_type.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].axhline(-ACCEPTABLE_DEGRADATION, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Strategy Type')
            axes[1, 1].set_ylabel('Average Degradation')
            axes[1, 1].set_title('Degradation by Strategy Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        # Identify robust selection criteria
        print("\n\nRobustness Analysis:")
        print("-" * 40)
        
        # What IS characteristics lead to better OOS?
        robust_mask = (comparison_df['degradation'] > -ACCEPTABLE_DEGRADATION) & (comparison_df['oos_sharpe'] > MIN_OOS_SHARPE)
        robust_strategies = comparison_df[robust_mask]
        
        if len(robust_strategies) > 0:
            print(f"Robust strategies: {len(robust_strategies)} ({len(robust_strategies)/len(comparison_df):.1%})")
            print(f"Average IS Sharpe of robust strategies: {robust_strategies['is_sharpe'].mean():.2f}")
            print(f"Average degradation of robust strategies: {robust_strategies['degradation'].mean():.1%}")
            
            # Compare with non-robust
            non_robust = comparison_df[~robust_mask]
            if len(non_robust) > 0:
                print(f"\nNon-robust strategies: {len(non_robust)}")
                print(f"Average IS Sharpe of non-robust: {non_robust['is_sharpe'].mean():.2f}")
                print(f"Average degradation of non-robust: {non_robust['degradation'].mean():.1%}")
                
                # Key insight
                if robust_strategies['is_sharpe'].mean() > non_robust['is_sharpe'].mean():
                    print("\n💡 Insight: Higher IS Sharpe correlates with OOS robustness")
                else:
                    print("\n💡 Insight: IS Sharpe alone doesn't predict OOS robustness")
        
        # Save comparison results
        comparison_df.to_csv('is_oos_comparison.csv', index=False)
        print("\n✅ Comparison results saved to is_oos_comparison.csv")
        
        # Store for further analysis
        is_oos_comparison = comparison_df
else:
    print("No window selection files found. Run window_selection.py first.")