# Regime Filtering Options for Strategy Analysis
import pandas as pd
import numpy as np

print("📊 REGIME FILTERING OPTIONS")
print("=" * 80)

print("\n1. VOLATILITY REGIMES")
print("-" * 40)
print("Available in: regime/volatility_regimes.py")
print("Splits market into Low/Medium/High volatility periods")
print("Usage: %run regime/volatility_regimes.py")
print("\nFilters:")
print("- Low Vol: Bottom 33% of rolling volatility")
print("- Medium Vol: Middle 33%")
print("- High Vol: Top 33%")

print("\n2. TREND REGIMES")
print("-" * 40)
print("Based on moving average relationships:")
print("- Uptrend: Price > SMA20 > SMA50")
print("- Downtrend: Price < SMA20 < SMA50")
print("- Sideways: Mixed signals")

print("\n3. TIME-BASED REGIMES")
print("-" * 40)
print("- Intraday: First/Last 30 minutes vs mid-day")
print("- Day of week: Monday effects, Friday effects")
print("- Monthly: Beginning/End of month")
print("- Seasonal: Quarterly patterns")

print("\n4. MARKET STRUCTURE REGIMES")
print("-" * 40)
print("- Volume regimes: High/Low volume periods")
print("- Spread regimes: Wide/Narrow bid-ask")
print("- Momentum regimes: Strong/Weak momentum")

print("\n5. COMBINED REGIME FILTERS")
print("-" * 40)
print("Example: High Vol + Downtrend = Crisis regime")
print("Example: Low Vol + Uptrend = Bull regime")

print("\n📈 COMPOUND SHARPE RATIO")
print("=" * 80)
print("Traditional Sharpe uses arithmetic mean returns")
print("Compound Sharpe uses geometric mean returns")
print("\nFormula comparison:")
print("Traditional: (mean of returns) / (std of returns) * sqrt(periods)")
print("Compound: (geometric mean) / (std of returns) * sqrt(periods)")
print("\nWhere geometric mean = (product(1+returns))^(1/n) - 1")

print("\n💡 WHY USE COMPOUND SHARPE?")
print("-" * 40)
print("1. Reflects actual investor experience (compound returns)")
print("2. Penalizes volatility drag appropriately")
print("3. Better for comparing strategies with different volatility")
print("4. Avoids the 'low volatility trap' in optimization")

print("\n🔧 IMPLEMENTATION EXAMPLE:")
print("-" * 40)
print('''
def compound_sharpe_ratio(returns, periods_per_year=252):
    """Calculate Sharpe using geometric mean instead of arithmetic"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    
    # Geometric mean
    compound_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    geometric_mean = (1 + compound_return)**(1/n_periods) - 1
    
    # Compound Sharpe
    return geometric_mean / returns.std() * np.sqrt(periods_per_year)
''')

print("\n📊 APPLYING REGIME FILTERS:")
print("-" * 40)
print("1. Load your analysis results")
print("2. Choose regime type (volatility, trend, etc.)")
print("3. Filter trades by regime")
print("4. Compare strategy performance across regimes")
print("5. Optimize stops/targets for each regime separately")

print("\n✅ RECOMMENDED WORKFLOW:")
print("1. Start with volatility regimes (most impactful)")
print("2. Add trend filters for directional strategies")
print("3. Consider time-based filters for intraday strategies")
print("4. Use compound Sharpe for final optimization")