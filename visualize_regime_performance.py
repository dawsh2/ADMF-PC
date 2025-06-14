#!/usr/bin/env python3
"""
Visualization tools for regime-based strategy performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_strategy_regime_heatmap(results_df: pd.DataFrame, classifier: str = None):
    """
    Create a heatmap showing strategy performance across regimes.
    """
    # Filter by classifier if specified
    if classifier:
        data = results_df[results_df['classifier'] == classifier]
        title = f"Strategy Performance Heatmap - {classifier}"
    else:
        data = results_df
        title = "Strategy Performance Heatmap - All Classifiers"
    
    # Create pivot table
    pivot = data.pivot_table(
        values='sharpe_ratio',
        index='strategy',
        columns='regime',
        aggfunc='mean'
    )
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Sharpe Ratio'}
    )
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_top_strategies_by_regime(results_df: pd.DataFrame, regime: str, top_n: int = 10):
    """
    Bar plot of top strategies for a specific regime.
    """
    regime_data = results_df[results_df['regime'] == regime].nlargest(top_n, 'sharpe_ratio')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sharpe ratio bars
    strategies = regime_data['strategy'] + '_' + regime_data['strategy_params'].str.replace('.parquet', '')
    ax1.barh(strategies, regime_data['sharpe_ratio'])
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_title(f'Top {top_n} Strategies - {regime} (by Sharpe)')
    ax1.grid(axis='x', alpha=0.3)
    
    # Win rate bars
    ax2.barh(strategies, regime_data['win_rate'])
    ax2.set_xlabel('Win Rate (%)')
    ax2.set_title(f'Top {top_n} Strategies - {regime} (by Win Rate)')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_distribution_pie(results_df: pd.DataFrame, classifier: str):
    """
    Pie chart showing regime distribution for a classifier.
    """
    # Get unique regime distributions (approximate from trade counts)
    regime_trades = results_df[
        results_df['classifier'] == classifier
    ].groupby('regime')['num_signals'].sum()
    
    plt.figure(figsize=(10, 8))
    plt.pie(
        regime_trades.values,
        labels=regime_trades.index,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(f'Regime Distribution - {classifier}')
    return plt.gcf()


def create_performance_report(results_df: pd.DataFrame, output_path: str = 'regime_performance_report.html'):
    """
    Create an HTML report with all visualizations and tables.
    """
    html_content = """
    <html>
    <head>
        <title>Strategy Performance by Regime Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .good { color: green; font-weight: bold; }
            .bad { color: red; font-weight: bold; }
            .neutral { color: black; }
        </style>
    </head>
    <body>
        <h1>Strategy Performance by Regime Analysis</h1>
    """
    
    # Add summary statistics
    html_content += "<h2>Summary Statistics</h2>"
    html_content += f"<p>Total Strategies Analyzed: {results_df['strategy'].nunique()}</p>"
    html_content += f"<p>Total Classifiers: {results_df['classifier'].nunique()}</p>"
    html_content += f"<p>Total Strategy-Regime Combinations: {len(results_df)}</p>"
    
    # Add top performers table
    html_content += "<h2>Top 20 Strategy-Regime Combinations</h2>"
    top_20 = results_df.nlargest(20, 'sharpe_ratio')
    
    html_content += "<table>"
    html_content += "<tr><th>Classifier</th><th>Regime</th><th>Strategy</th><th>Sharpe</th><th>Avg Return</th><th>Win Rate</th><th>Trades</th></tr>"
    
    for _, row in top_20.iterrows():
        sharpe_class = 'good' if row['sharpe_ratio'] > 1 else 'neutral' if row['sharpe_ratio'] > 0 else 'bad'
        win_class = 'good' if row['win_rate'] > 55 else 'neutral' if row['win_rate'] > 45 else 'bad'
        
        html_content += f"""
        <tr>
            <td>{row['classifier']}</td>
            <td>{row['regime']}</td>
            <td>{row['strategy']}</td>
            <td class='{sharpe_class}'>{row['sharpe_ratio']:.3f}</td>
            <td>{row['avg_return_pct']:.4f}%</td>
            <td class='{win_class}'>{row['win_rate']:.1f}%</td>
            <td>{row['num_signals']}</td>
        </tr>
        """
    
    html_content += "</table>"
    
    # Add regime-specific sections
    for classifier in results_df['classifier'].unique():
        html_content += f"<h2>Classifier: {classifier}</h2>"
        classifier_data = results_df[results_df['classifier'] == classifier]
        
        for regime in sorted(classifier_data['regime'].unique()):
            regime_data = classifier_data[classifier_data['regime'] == regime]
            top_5 = regime_data.nlargest(5, 'sharpe_ratio')
            
            html_content += f"<h3>Regime: {regime}</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Strategy</th><th>Parameters</th><th>Sharpe</th><th>Win Rate</th></tr>"
            
            for _, row in top_5.iterrows():
                html_content += f"""
                <tr>
                    <td>{row['strategy']}</td>
                    <td>{row['strategy_params']}</td>
                    <td>{row['sharpe_ratio']:.3f}</td>
                    <td>{row['win_rate']:.1f}%</td>
                </tr>
                """
            
            html_content += "</table>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to {output_path}")


def quick_performance_summary(results_df: pd.DataFrame):
    """
    Print a quick summary of performance across all regimes.
    """
    print("QUICK PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Best strategy for each regime
    print("\nBest Strategy per Regime:")
    best_per_regime = results_df.loc[results_df.groupby('regime')['sharpe_ratio'].idxmax()]
    
    for _, row in best_per_regime.iterrows():
        print(f"{row['regime']:<20} → {row['strategy']:<30} (Sharpe: {row['sharpe_ratio']:.3f})")
    
    # Most consistent strategies
    print("\n\nMost Consistent Strategies (across all regimes):")
    consistency = results_df.groupby('strategy').agg({
        'sharpe_ratio': ['mean', 'std', 'count']
    }).round(3)
    consistency.columns = ['avg_sharpe', 'std_sharpe', 'num_regimes']
    consistency = consistency[consistency['num_regimes'] >= 5]  # Active in at least 5 regimes
    consistency['consistency_score'] = consistency['avg_sharpe'] / (consistency['std_sharpe'] + 0.1)
    
    top_consistent = consistency.nlargest(10, 'consistency_score')
    print(top_consistent)
    
    # Strategies that work in opposite regimes
    print("\n\nVersatile Strategies (work in opposite regimes):")
    
    # Find strategies that work in both uptrend and downtrend
    uptrend_strategies = set(results_df[
        (results_df['regime'].str.contains('up')) & 
        (results_df['sharpe_ratio'] > 0.5)
    ]['strategy'])
    
    downtrend_strategies = set(results_df[
        (results_df['regime'].str.contains('down')) & 
        (results_df['sharpe_ratio'] > 0.5)
    ]['strategy'])
    
    versatile = uptrend_strategies.intersection(downtrend_strategies)
    print(f"Found {len(versatile)} strategies that work in both up and down trends:")
    for s in list(versatile)[:10]:
        print(f"  - {s}")


if __name__ == "__main__":
    # Example usage
    results_df = pd.read_csv('strategy_regime_performance.csv')
    
    # Quick summary
    quick_performance_summary(results_df)
    
    # Create visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Heatmap for each classifier
    for classifier in results_df['classifier'].unique():
        fig = plot_strategy_regime_heatmap(results_df, classifier)
        fig.savefig(f'heatmap_{classifier}.png', dpi=150, bbox_inches='tight')
    
    # Top strategies for each regime
    for regime in results_df['regime'].unique():
        fig = plot_top_strategies_by_regime(results_df, regime)
        fig.savefig(f'top_strategies_{regime}.png', dpi=150, bbox_inches='tight')
    
    # Create HTML report
    create_performance_report(results_df)