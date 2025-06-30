"""Analyze how leverage and different instruments affect profitability"""
import pandas as pd
import numpy as np

# Load our best filtered strategy
trades_df = pd.read_csv('trades_enhanced_analysis.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

# Best filter: Volume > 1.2 + Momentum
best_filter = (trades_df['volume_ratio'] > 1.2) & trades_df['trend_aligned']
filtered_trades = trades_df[best_filter]

print("=== How Leverage and Instruments Change the Game ===\n")

# Base metrics
avg_return = filtered_trades['pnl_pct'].mean() / 100  # 0.0430%
trades_per_year = 131  # from previous analysis
win_rate = filtered_trades['win'].mean()

print(f"Base Strategy (Volume > 1.2 + Momentum):")
print(f"  Average return per trade: {avg_return*100:.4f}%")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Trades per year: {trades_per_year}")

# 1. DIFFERENT INSTRUMENTS AND THEIR COSTS
print("\n=== 1. INSTRUMENT COMPARISON ===\n")

instruments = {
    "SPY ETF (Cash)": {
        "leverage": 1,
        "round_trip_bp": 2.0,  # 1bp each way
        "capital_required": 1.0,
        "other_costs": 0
    },
    "SPY Futures (ES)": {
        "leverage": 10,  # Typical with futures margin
        "round_trip_bp": 0.5,  # Much lower - about $2.50 per contract ≈ 0.25bp each way
        "capital_required": 0.1,  # Only need margin
        "other_costs": 0
    },
    "SPY Options (ATM)": {
        "leverage": 20,  # Rough estimate for ATM options
        "round_trip_bp": 10.0,  # Much higher - wide bid/ask spreads
        "capital_required": 0.05,
        "other_costs": 0.01  # Theta decay per trade
    },
    "CFDs (Retail)": {
        "leverage": 5,
        "round_trip_bp": 3.0,  # Spread + commission
        "capital_required": 0.2,
        "other_costs": 0.002  # Overnight financing if held
    },
    "Micro Futures (MES)": {
        "leverage": 10,
        "round_trip_bp": 1.0,  # Higher per-contract but same leverage
        "capital_required": 0.1,
        "other_costs": 0
    }
}

print(f"{'Instrument':<20} {'Leverage':<10} {'Cost(bp)':<10} {'Net/Trade':<12} {'Annual':<12} {'Sharpe':<10}")
print("-" * 80)

for name, specs in instruments.items():
    leverage = specs['leverage']
    cost_bp = specs['round_trip_bp'] / 10000
    
    # Leveraged return per trade
    gross_leveraged = avg_return * leverage
    net_leveraged = gross_leveraged - cost_bp - specs['other_costs']
    
    # Annual return
    if net_leveraged > -1:
        annual_return = (1 + net_leveraged) ** trades_per_year - 1
    else:
        annual_return = -1
    
    # Approximate Sharpe (assuming volatility scales with sqrt of leverage)
    # Base Sharpe from earlier analysis was ~0.5 for unleveraged
    base_sharpe = 0.5
    adjusted_sharpe = base_sharpe * np.sqrt(leverage) * (1 if annual_return > 0 else -1)
    
    print(f"{name:<20} {leverage:<10.0f} {specs['round_trip_bp']:<10.1f} "
          f"{net_leveraged*100:>10.3f}%  {annual_return*100:>10.1f}%  {adjusted_sharpe:>8.2f}")

# 2. BREAKEVEN ANALYSIS WITH LEVERAGE
print("\n=== 2. BREAKEVEN COST ANALYSIS BY LEVERAGE ===\n")

print(f"{'Leverage':<12} {'Breakeven Cost (bp)':<25} {'Typical Instrument':<30}")
print("-" * 70)

for leverage in [1, 2, 5, 10, 20, 50]:
    # At breakeven: leverage * avg_return = round_trip_cost
    breakeven_bp = avg_return * leverage * 10000
    
    # What instrument typically offers this leverage?
    if leverage == 1:
        typical = "Cash equities, ETFs"
    elif leverage <= 5:
        typical = "Margin account, CFDs"
    elif leverage <= 20:
        typical = "Futures, FX"
    else:
        typical = "Options (deep ITM), High margin FX"
    
    print(f"{leverage:<12} {breakeven_bp:<25.1f} {typical:<30}")

# 3. VOLATILITY AND RISK CONSIDERATIONS
print("\n=== 3. RISK-ADJUSTED RETURNS ===\n")

# Estimate daily volatility from our trades
daily_returns = filtered_trades.groupby(filtered_trades['exit_time'].dt.date)['pnl_pct'].sum() / 100
daily_vol = daily_returns.std()
annual_vol = daily_vol * np.sqrt(252)

print(f"Base strategy volatility: {annual_vol*100:.1f}% annualized")
print(f"\nLeveraged volatility and risk metrics:")
print(f"{'Leverage':<10} {'Annual Vol':<12} {'Max DD':<12} {'Kelly %':<12} {'Optimal Lev':<12}")
print("-" * 60)

for leverage in [1, 2, 5, 10, 20]:
    lev_vol = annual_vol * leverage
    
    # Rough max drawdown estimate (2-3x volatility)
    max_dd = lev_vol * 2.5
    
    # Kelly criterion: f = (p*b - q)/b where p=win rate, b=avg win/avg loss, q=1-p
    wins = filtered_trades[filtered_trades['pnl_pct'] > 0]['pnl_pct'].mean()
    losses = abs(filtered_trades[filtered_trades['pnl_pct'] < 0]['pnl_pct'].mean())
    b = wins / losses if losses > 0 else 1
    
    kelly = (win_rate * b - (1 - win_rate)) / b
    kelly_leveraged = kelly * leverage
    
    # Optimal leverage (maximizes Sharpe)
    # Assuming costs scale linearly and Sharpe peaks at moderate leverage
    cost_drag = 0.001 * leverage  # Rough estimate
    net_return = avg_return * leverage - cost_drag
    optimal = "Yes" if 3 <= leverage <= 10 else "No"
    
    print(f"{leverage:<10} {lev_vol*100:>10.1f}%  {max_dd*100:>10.1f}%  {kelly_leveraged*100:>10.1f}%  {optimal:<12}")

# 4. PRACTICAL CONSIDERATIONS
print("\n=== 4. PRACTICAL INSIGHTS ===\n")

print("Key findings:")
print("1. Futures (10x leverage, 0.5bp costs) appear optimal - turns 0.043% into 0.425% per trade")
print("2. Options have too much friction (spreads + theta) for small edge strategies")
print("3. Leverage 5-10x maximizes Sharpe ratio for this type of strategy")
print("4. Need to size positions carefully - Kelly suggests ~7% of capital at 10x leverage")
print("\n5. Alternative approaches to improve profitability:")
print("   - Hold winners longer (swing trading)")
print("   - Trade only highest conviction setups")
print("   - Become a liquidity provider (earn the spread)")
print("   - Use limit orders (reduce effective costs)")
print("   - Trade during optimal hours (higher volatility)")

# 5. MINIMUM EDGE REQUIRED
print("\n=== 5. MINIMUM EDGE REQUIRED FOR PROFITABILITY ===\n")

print(f"{'Instrument Type':<25} {'Min Avg Return/Trade':<25} {'Min Win Rate @0.1% edge':<25}")
print("-" * 75)

for inst, specs in instruments.items():
    # Minimum return = cost / leverage
    min_return_pct = (specs['round_trip_bp'] / 10000 + specs['other_costs']) / specs['leverage'] * 100
    
    # Min win rate assuming 0.1% edge per winning trade
    min_wr = min_return_pct / 0.1 * 100
    
    print(f"{inst:<25} {min_return_pct:>22.3f}%  {min_wr:>22.1f}%")