"""Analyze performance with zero execution costs"""
import pandas as pd
import numpy as np

# Load the trades
trades_df = pd.read_csv('trades_enhanced_analysis.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("=== Performance Analysis: With and Without Execution Costs ===\n")

# Calculate base metrics
date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
base_tpd = len(trades_df) / date_range_days

# Original performance
print("ORIGINAL STRATEGY (All 851 trades):")
print(f"Average return per trade: {trades_df['pnl_pct'].mean():.4f}%")
print(f"Trades per day: {base_tpd:.2f}")
print(f"Trades per year: {base_tpd * 252:.0f}")

# Calculate annual returns
avg_return_decimal = trades_df['pnl_pct'].mean() / 100
trades_per_year = base_tpd * 252

# With 1bp cost
exec_cost_1bp = 0.0001
net_return_1bp = avg_return_decimal - (2 * exec_cost_1bp)
annual_net_1bp = (1 + net_return_1bp) ** trades_per_year - 1

# With 0.5bp cost  
exec_cost_05bp = 0.00005
net_return_05bp = avg_return_decimal - (2 * exec_cost_05bp)
annual_net_05bp = (1 + net_return_05bp) ** trades_per_year - 1

# Zero cost
annual_gross = (1 + avg_return_decimal) ** trades_per_year - 1

print(f"\nAnnualized Returns:")
print(f"  No cost: {annual_gross*100:.1f}%")
print(f"  0.5bp cost: {annual_net_05bp*100:.1f}%")
print(f"  1bp cost: {annual_net_1bp*100:.1f}%")
print(f"  2bp cost: {((1 + avg_return_decimal - 4*exec_cost_1bp) ** trades_per_year - 1)*100:.1f}%")

# Now for filtered strategies
print("\n" + "="*60)
print("\nFILTERED STRATEGIES:\n")

# Define filters
filters = [
    ("Volume > 1.2", trades_df['volume_ratio'] > 1.2),
    ("Volume > 1.0", trades_df['volume_ratio'] > 1.0),
    ("Volume > 1.2 + Momentum", 
     (trades_df['volume_ratio'] > 1.2) & trades_df['trend_aligned']),
    ("Volume > 1.0 + Momentum", 
     (trades_df['volume_ratio'] > 1.0) & trades_df['trend_aligned']),
]

for name, mask in filters:
    filtered = trades_df[mask]
    if len(filtered) > 0:
        avg_ret = filtered['pnl_pct'].mean()
        filter_ratio = len(filtered) / len(trades_df)
        tpd = base_tpd * filter_ratio
        tpy = tpd * 252
        
        print(f"{name}:")
        print(f"  Trades: {len(filtered)} ({len(filtered)/len(trades_df)*100:.1f}%)")
        print(f"  Win Rate: {filtered['win'].mean()*100:.1f}%")
        print(f"  Avg return/trade: {avg_ret:.4f}%")
        print(f"  Trades per year: {tpy:.0f}")
        
        # Calculate returns with different costs
        avg_ret_decimal = avg_ret / 100
        
        # No cost
        annual_0 = (1 + avg_ret_decimal) ** tpy - 1
        # 0.5bp
        annual_05 = (1 + avg_ret_decimal - 0.0001) ** tpy - 1
        # 1bp
        annual_1 = (1 + avg_ret_decimal - 0.0002) ** tpy - 1
        # 2bp
        annual_2 = (1 + avg_ret_decimal - 0.0004) ** tpy - 1
        
        print(f"  Annual returns:")
        print(f"    No cost: {annual_0*100:.1f}%")
        print(f"    0.5bp: {annual_05*100:.1f}%") 
        print(f"    1bp: {annual_1*100:.1f}%")
        print(f"    2bp: {annual_2*100:.1f}%")
        print()

# Break-even analysis
print("="*60)
print("\nBREAK-EVEN ANALYSIS:\n")

print("Execution cost needed to break even (0% annual return):")

for name, mask in filters:
    filtered = trades_df[mask]
    if len(filtered) > 0:
        avg_ret = filtered['pnl_pct'].mean() / 100
        filter_ratio = len(filtered) / len(trades_df)
        tpy = base_tpd * filter_ratio * 252
        
        # Solve for break-even: (1 + avg_ret - 2*cost)^tpy = 1
        # This means avg_ret - 2*cost = 0
        breakeven_cost_per_side = avg_ret / 2
        breakeven_bp = breakeven_cost_per_side * 10000
        
        print(f"{name}: {breakeven_bp:.2f}bp per side ({breakeven_bp*2:.2f}bp round trip)")