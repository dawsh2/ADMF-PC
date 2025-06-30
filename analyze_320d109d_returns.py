"""Calculate annual returns for workspace 320d109d strategies"""
import numpy as np

print("=== ANNUAL RETURN ANALYSIS: WORKSPACE 320d109d ===\n")

# Top strategies from analysis
strategies = [
    {"id": 80, "trades_year": 3692, "edge_bps": 0.31, "desc": "Best Annual BPS"},
    {"id": 88, "trades_year": 1695, "edge_bps": 0.62, "desc": "Highest Edge"}, 
    {"id": 40, "trades_year": 3911, "edge_bps": 0.26, "desc": "Most Active"},
    {"id": 81, "trades_year": 3297, "edge_bps": 0.27, "desc": "Second Best Annual"},
    {"id": 48, "trades_year": 1516, "edge_bps": 0.50, "desc": "Balanced"}
]

print("ANNUAL RETURNS WITH EXECUTION COSTS")
print("=" * 80)
print("ID  | Description      | Trades/Yr | Edge | 0bp   | 0.25bp | 0.5bp | 0.75bp | 1bp")
print("-" * 80)

for s in strategies:
    returns = []
    for cost_bps in [0, 0.25, 0.5, 0.75, 1.0]:
        net_edge = s['edge_bps'] - cost_bps
        if net_edge > 0:
            annual_return = (1 + net_edge/10000) ** s['trades_year'] - 1
            returns.append(f"{annual_return*100:5.1f}%")
        else:
            returns.append("  NEG ")
    
    print(f"{s['id']:3} | {s['desc']:16} | {s['trades_year']:9.0f} | {s['edge_bps']:4.2f} | " + " | ".join(returns))

print("\n" + "=" * 80)
print("\nCOMPARISON TO 1-MINUTE RESULTS")
print("=" * 80)

print("\n1-Minute (High Vol Filter):")
print("- 148 trades/month = 1,776 trades/year")
print("- 1.26 bps edge")
print("- Annual returns: 25.1% (0bp), 14.5% (0.5bp), 4.7% (1bp)")

print("\n5-Minute Best (Strategy 80):")
print("- 3,692 trades/year")
print("- 0.31 bps edge")
print("- Annual returns: 12.0% (0bp), 2.8% (0.25bp), NEG (0.5bp)")

print("\n5-Minute High Edge (Strategy 88):")
print("- 1,695 trades/year") 
print("- 0.62 bps edge")
print("- Annual returns: 11.1% (0bp), 6.5% (0.25bp), 2.1% (0.5bp)")

print("\nKEY INSIGHTS:")
print("-" * 50)
print("1. This 5m version trades MUCH more frequently")
print("2. Edge per trade is lower but total opportunity is higher")
print("3. More sensitive to execution costs due to lower edge")
print("4. Strategy 88 offers best balance (0.62 bps × 1,695 trades)")

print("\n\nOPTIMAL PORTFOLIO COMBINATION")
print("=" * 80)
print("\nCombining filtered strategies on 5-minute:")
print("- Bollinger RSI: 15-19 bps edge, ~130 trades/year")
print("- Swing Pivot (Strategy 88): 0.62 bps edge, 1,695 trades/year")

# Portfolio calculation
bb_trades = 130
bb_edge = 17  # midpoint of 15-19
sp_trades = 1695
sp_edge = 0.62

total_trades = bb_trades + sp_trades
weighted_edge = (bb_trades * bb_edge + sp_trades * sp_edge) / total_trades

print(f"\nPortfolio metrics:")
print(f"- Total trades/year: {total_trades:,}")
print(f"- Weighted avg edge: {weighted_edge:.2f} bps")
print(f"- Bollinger weight: {bb_trades/total_trades*100:.1f}%")
print(f"- Swing Pivot weight: {sp_trades/total_trades*100:.1f}%")

print(f"\nExpected annual returns:")
for cost in [0.5, 1, 2]:
    bb_net = bb_edge - cost
    sp_net = sp_edge - cost
    
    if bb_net > 0:
        bb_return = (1 + bb_net/10000) ** bb_trades - 1
    else:
        bb_return = 0
        
    if sp_net > 0:
        sp_return = (1 + sp_net/10000) ** sp_trades - 1
    else:
        sp_return = 0
    
    # Weight by trade count for combined return
    combined = (bb_return * bb_trades + sp_return * sp_trades) / total_trades
    
    print(f"- {cost} bps cost: {combined*100:.1f}% combined "
          f"(BB: {bb_return*100:.1f}%, SP: {sp_return*100:.1f}%)")