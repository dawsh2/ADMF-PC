#!/usr/bin/env python3
"""
Visualize the impact of execution costs on the DuckDB ensemble strategy.

Creates plots showing gross vs net performance curves and trade-by-trade cost impact.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from apply_execution_costs import calculate_net_log_return_pnl, COMMISSION_PER_SHARE, SLIPPAGE_PCT

def plot_performance_comparison(results, period_name, save_path=None):
    """Create comprehensive performance comparison plots."""
    
    gross_trades = results['gross']['trades']
    net_trades = results['net']['trades']
    
    if not gross_trades:
        print(f"No trades found for {period_name}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Execution Cost Impact Analysis - {period_name}', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Return Curves
    ax1 = axes[0, 0]
    
    # Calculate cumulative returns
    gross_cumret = np.cumsum([t['log_return'] for t in gross_trades])
    net_cumret = np.cumsum([t['log_return'] for t in net_trades])
    
    # Convert to percentage for plotting
    gross_cumret_pct = np.exp(gross_cumret) - 1
    net_cumret_pct = np.exp(net_cumret) - 1
    
    trade_numbers = range(1, len(gross_trades) + 1)
    
    ax1.plot(trade_numbers, gross_cumret_pct * 100, label='Gross Returns', color='green', linewidth=2)
    ax1.plot(trade_numbers, net_cumret_pct * 100, label='Net Returns', color='red', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Cumulative Return Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Trade Return Distribution
    ax2 = axes[0, 1]
    
    gross_returns_pct = [(np.exp(t['log_return']) - 1) * 100 for t in gross_trades]
    net_returns_pct = [(np.exp(t['log_return']) - 1) * 100 for t in net_trades]
    
    ax2.hist(gross_returns_pct, bins=50, alpha=0.7, label='Gross Returns', color='green', density=True)
    ax2.hist(net_returns_pct, bins=50, alpha=0.7, label='Net Returns', color='red', density=True)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trade Return (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Trade Return Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Execution Cost per Trade
    ax3 = axes[1, 0]
    
    execution_costs_pct = [t.get('execution_cost_pct', 0) * 100 for t in net_trades]
    
    ax3.plot(trade_numbers, execution_costs_pct, color='orange', linewidth=1, alpha=0.7)
    ax3.axhline(y=np.mean(execution_costs_pct), color='red', linestyle='--', 
                label=f'Average: {np.mean(execution_costs_pct):.3f}%')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Execution Cost (%)')
    ax3.set_title('Execution Cost per Trade')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Win Rate Analysis
    ax4 = axes[1, 1]
    
    # Calculate rolling win rates
    window = 100
    if len(gross_trades) >= window:
        gross_win_rates = []
        net_win_rates = []
        
        for i in range(window, len(gross_trades) + 1):
            gross_window = gross_trades[i-window:i]
            net_window = net_trades[i-window:i]
            
            gross_wins = sum(1 for t in gross_window if t['log_return'] > 0)
            net_wins = sum(1 for t in net_window if t['log_return'] > 0)
            
            gross_win_rates.append(gross_wins / window * 100)
            net_win_rates.append(net_wins / window * 100)
        
        win_rate_trades = range(window, len(gross_trades) + 1)
        
        ax4.plot(win_rate_trades, gross_win_rates, label='Gross Win Rate', color='green', linewidth=2)
        ax4.plot(win_rate_trades, net_win_rates, label='Net Win Rate', color='red', linewidth=2)
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Break-even')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title(f'Rolling Win Rate ({window}-trade window)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, f'Insufficient trades for rolling analysis\n(need {window}, have {len(gross_trades)})', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Rolling Win Rate Analysis')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def create_cost_breakdown_chart(results, period_name, save_path=None):
    """Create a detailed breakdown of execution costs."""
    
    if not results['execution_costs']:
        print(f"No execution cost data for {period_name}")
        return
    
    exec_costs = results['execution_costs']
    
    # Calculate cost components
    avg_commission_pct = np.mean([ec['commission_cost_pct'] * 2 for ec in exec_costs]) * 100  # Entry + exit
    avg_slippage_pct = np.mean([ec['slippage_cost_pct'] * 2 for ec in exec_costs]) * 100  # Entry + exit
    avg_total_cost_pct = np.mean([ec['total_execution_cost_pct'] for ec in exec_costs]) * 100
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Execution Cost Breakdown - {period_name}', fontsize=14, fontweight='bold')
    
    # Pie chart of cost components
    costs = [avg_commission_pct, avg_slippage_pct]
    labels = ['Commission\n(Entry + Exit)', 'Slippage\n(Entry + Exit)']
    colors = ['#ff9999', '#66b3ff']
    
    wedges, texts, autotexts = ax1.pie(costs, labels=labels, colors=colors, autopct='%1.3f%%', 
                                       startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Average Cost per Trade')
    
    # Bar chart comparing gross vs net performance
    metrics = ['Total Return', 'Win Rate', 'Avg Trade Return']
    gross_values = [
        results['gross']['percentage_return'] * 100,
        results['gross']['win_rate'] * 100, 
        (np.exp(results['gross']['avg_trade_log_return']) - 1) * 100
    ]
    net_values = [
        results['net']['percentage_return'] * 100,
        results['net']['win_rate'] * 100,
        (np.exp(results['net']['avg_trade_log_return']) - 1) * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, gross_values, width, label='Gross', color='green', alpha=0.7)
    ax2.bar(x + width/2, net_values, width, label='Net', color='red', alpha=0.7)
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Value (%)')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for i, (g, n) in enumerate(zip(gross_values, net_values)):
        ax2.text(i - width/2, g + (max(gross_values) * 0.01), f'{g:.1f}%', 
                ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, n + (max(gross_values) * 0.01), f'{n:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost breakdown chart saved to: {save_path}")
    
    plt.show()

def create_summary_table(results):
    """Create a formatted summary table of results."""
    
    print("\n" + "="*100)
    print("EXECUTION COST IMPACT SUMMARY TABLE")
    print("="*100)
    
    # Create DataFrame for clean formatting
    summary_data = []
    
    for period, period_results in results.items():
        gross = period_results['gross']
        net = period_results['net']
        
        summary_data.append({
            'Period': period.replace('_', ' ').title(),
            'Gross Return (%)': f"{gross['percentage_return']:.2%}",
            'Net Return (%)': f"{net['percentage_return']:.2%}",
            'Cost Impact (%)': f"{-(gross['percentage_return'] - net['percentage_return']):.2%}",
            'Gross Win Rate (%)': f"{gross['win_rate']:.1%}",
            'Net Win Rate (%)': f"{net['win_rate']:.1%}",
            'Trades': f"{gross['num_trades']:,}",
            'Avg Duration (bars)': f"{np.mean([t['bars_held'] for t in gross['trades']]):.1f}" if gross['trades'] else "0.0"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Cost breakdown
    print(f"\n{'='*60}")
    print("EXECUTION COST COMPONENTS")
    print(f"{'='*60}")
    print(f"Commission per Share:       ${COMMISSION_PER_SHARE:.3f}")
    print(f"Slippage (basis points):    {SLIPPAGE_PCT * 10000:.0f} bps")
    print(f"Total Cost per Trade:       ~{(SLIPPAGE_PCT * 2 + 0.0017) * 100:.3f}% (varies by price)")
    print(f"Application:                Entry + Exit (doubled)")

def main():
    # Path to the signal trace file
    signal_file = Path("traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    
    if not signal_file.exists():
        print(f"Signal file not found: {signal_file}")
        return
    
    print(f"Loading and processing data from: {signal_file}")
    df = pd.read_parquet(signal_file)
    
    # Map column names
    df = df.rename(columns={
        'idx': 'bar_idx',
        'px': 'price', 
        'val': 'signal_value'
    })
    
    # Sort by bar_idx
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    # Analyze both periods
    results = {}
    
    # Full period
    print("Analyzing full period...")
    full_results = calculate_net_log_return_pnl(df)
    results['full_period'] = full_results
    
    # Last 12k bars
    print("Analyzing last 12k bars...")
    max_bar = df['bar_idx'].max()
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    last_12k_results = calculate_net_log_return_pnl(last_12k_df)
    results['last_12k_bars'] = last_12k_results
    
    # Create output directory
    output_dir = Path("execution_cost_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots for each period
    for period, period_results in results.items():
        print(f"\nCreating plots for {period}...")
        
        # Performance comparison plots
        plot_path = output_dir / f"performance_comparison_{period}.png"
        plot_performance_comparison(period_results, period.replace('_', ' ').title(), plot_path)
        
        # Cost breakdown chart
        cost_path = output_dir / f"cost_breakdown_{period}.png"
        create_cost_breakdown_chart(period_results, period.replace('_', ' ').title(), cost_path)
    
    # Print summary table
    create_summary_table(results)
    
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()