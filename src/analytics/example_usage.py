"""
Example usage of the ADMF-PC Analytics Library

This script demonstrates the key features of the analytics library.
For interactive use, run these commands in a Jupyter notebook.
"""

from pathlib import Path

# Example 1: Quick Start
print("=== Example 1: Quick Start ===")
print("""
from analytics import quick_analysis

# Load latest results
ta, pd = quick_analysis('config/keltner/results/latest')

# Get strategy summary
summary = ta.summary()
print(summary)
""")

# Example 2: Filter Analysis
print("\n=== Example 2: Filter Analysis ===")
print("""
# Compare filtered vs unfiltered strategies
filter_comparison = ta.compare_filters()
print(filter_comparison)

# Analyze specific filter types
volatility_filters = ta.sql('''
    SELECT strategy_id, COUNT(*) as signals 
    FROM signals 
    WHERE strategy_id IN (
        SELECT strategy_id 
        FROM strategy_summary 
        WHERE total_signals < 3000
    )
    GROUP BY strategy_id
''').df()
""")

# Example 3: Pattern Discovery
print("\n=== Example 3: Pattern Discovery ===")
print("""
# Find common signal sequences
sequences = pd.find_signal_sequences(min_frequency=0.02)
print(f"Found {len(sequences)} common patterns")

# Find profitable market conditions
conditions = pd.find_profitable_conditions(min_sharpe=1.5)
print(conditions)

# Get analysis suggestions
suggestions = pd.suggest_explorations()
for s in suggestions:
    print(f"- {s['title']}: {s['action']}")
""")

# Example 4: Trade Analysis
print("\n=== Example 4: Trade Analysis ===")
print("""
from analytics import TradeAnalyzer

# Extract trades from signals
trades_df = ta.get_trades(min_duration=5)
print(f"Found {len(trades_df)} trades")

# Analyze trade performance
analyzer = TradeAnalyzer(ta.conn.execute("SELECT * FROM signals").df())
stats = analyzer.summary_stats()
print(f"Win Rate: {stats['win_rate']:.1%}")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
print(f"Profit Factor: {stats['profit_factor']:.2f}")
""")

# Example 5: Custom Queries
print("\n=== Example 5: Custom SQL Queries ===")
print("""
# Find strategies with highest signal concentration
query = '''
    WITH signal_density AS (
        SELECT 
            strategy_id,
            COUNT(*) as total_signals,
            MAX(bar_idx) - MIN(bar_idx) as bar_range,
            COUNT(*)::FLOAT / (MAX(bar_idx) - MIN(bar_idx) + 1) as density
        FROM signals
        GROUP BY strategy_id
    )
    SELECT * 
    FROM signal_density
    ORDER BY density DESC
    LIMIT 10
'''
high_density = ta.sql(query)
print(high_density)
""")

# Example 6: Save Patterns
print("\n=== Example 6: Save Discovered Patterns ===")
print("""
# Create a pattern from a successful query
pattern = pd.create_pattern(
    name="High Volatility Momentum",
    description="Strategies that perform well in high volatility regimes",
    query='''
        SELECT s.strategy_id, COUNT(*) as signals
        FROM signals s
        WHERE s.bar_idx IN (
            SELECT bar_idx FROM market_regimes WHERE volatility = 'HIGH'
        )
        GROUP BY s.strategy_id
        HAVING COUNT(*) > 100
    ''',
    tags=['volatility', 'momentum']
)

# Save to library
pattern_id = pd.save_pattern(pattern)
print(f"Saved pattern: {pattern_id}")

# Search patterns later
momentum_patterns = pd.library.search(tags=['momentum'])
""")

# Example 7: Performance Analysis
print("\n=== Example 7: Performance by Filter Type ===")
print("""
# Hypothetical performance by filter
perf_by_filter = ta.performance_by_filter()
print(perf_by_filter)

# Export for external analysis
export_path = ta.export_for_mining()
print(f"Exported to {export_path}")
""")

print("\n=== Interactive Workflow ===")
print("""
# Best used in Jupyter:
# 1. Start with quick_analysis() for overview
# 2. Use ta.sql() for custom exploration  
# 3. Discover patterns with pd methods
# 4. Save interesting findings to pattern library
# 5. Export data for deeper analysis

# The library returns DataFrames for easy plotting:
# ta.summary().plot.scatter('total_signals', 'filter')
# ta.get_trades()['return_pct'].hist(bins=50)
""")

if __name__ == "__main__":
    print("\nThis script shows example usage.")
    print("For actual analysis, run these commands in a Jupyter notebook or Python REPL.")
    print("\nTo get started:")
    print("  $ python")
    print("  >>> from analytics import quick_analysis")
    print("  >>> ta, pd = quick_analysis('config/keltner/results/latest')")