import pandas as pd
import numpy as np

# Read positions close
positions_close = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_close/positions_close.parquet')

# Extract P&L values
pnls = []
exit_types = []
for i in range(len(positions_close)):
    metadata = positions_close.iloc[i]['metadata']
    if isinstance(metadata, dict):
        pnl = metadata.get('realized_pnl', 0)
        exit_type = metadata.get('metadata', {}).get('exit_type', 'unknown')
        pnls.append(pnl)
        exit_types.append(exit_type)

pnls = np.array(pnls)
exit_types = np.array(exit_types)

# Analyze P&L distribution
print("=== P&L Distribution ===")
print(f"Count: {len(pnls)}")
print(f"Mean P&L: ${np.mean(pnls):.4f}")
print(f"Median P&L: ${np.median(pnls):.4f}")
print(f"Std Dev: ${np.std(pnls):.4f}")
print(f"Min P&L: ${np.min(pnls):.4f}")
print(f"Max P&L: ${np.max(pnls):.4f}")

# By exit type
print("\n=== P&L by Exit Type ===")
for exit_type in ['stop_loss', 'take_profit', 'signal']:
    mask = exit_types == exit_type
    if np.any(mask):
        type_pnls = pnls[mask]
        print(f"\n{exit_type}:")
        print(f"  Count: {len(type_pnls)}")
        print(f"  Mean: ${np.mean(type_pnls):.4f}")
        print(f"  Total: ${np.sum(type_pnls):.2f}")
        print(f"  Min: ${np.min(type_pnls):.4f}")
        print(f"  Max: ${np.max(type_pnls):.4f}")

# Check if we're hitting targets correctly
print("\n=== Target Analysis ===")
# Expected values based on $1 trades
expected_stop_loss = -0.00075  # -0.075%
expected_take_profit = 0.0015   # 0.15%

stop_losses = pnls[exit_types == 'stop_loss']
take_profits = pnls[exit_types == 'take_profit']

if len(stop_losses) > 0:
    print(f"Stop losses: {len(stop_losses)} trades")
    print(f"  Expected P&L per trade: ${expected_stop_loss:.4f}")
    print(f"  Actual mean P&L: ${np.mean(stop_losses):.4f}")
    print(f"  Actual values range: ${np.min(stop_losses):.4f} to ${np.max(stop_losses):.4f}")

if len(take_profits) > 0:
    print(f"\nTake profits: {len(take_profits)} trades")
    print(f"  Expected P&L per trade: ${expected_take_profit:.4f}")
    print(f"  Actual mean P&L: ${np.mean(take_profits):.4f}")
    print(f"  Actual values range: ${np.min(take_profits):.4f} to ${np.max(take_profits):.4f}")

# Calculate returns
total_pnl = np.sum(pnls)
initial_capital = 100000
position_size = 1  # Assuming $1 per trade

print(f"\n=== Returns Calculation ===")
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Number of trades: {len(pnls)}")
print(f"Initial capital: ${initial_capital:,.0f}")
print(f"Final capital: ${initial_capital + total_pnl:,.2f}")
print(f"Return: {(total_pnl / initial_capital) * 100:.4f}%")

# Check if position sizing might be the issue
print(f"\n=== Position Sizing Check ===")
print(f"If each trade used full capital:")
print(f"  Return would be: {(total_pnl / len(pnls)) * 100:.4f}% per trade")
print(f"  Total return: {(total_pnl / len(pnls)) * len(pnls) * 100:.4f}%")

# Calculate what position size would give 10.27% returns
target_return = 0.1027
required_position_size = (target_return * initial_capital) / total_pnl
print(f"\nTo achieve 10.27% returns with current P&L:")
print(f"  Required position size: ${required_position_size:.2f} per trade")
print(f"  As % of capital: {(required_position_size / initial_capital) * 100:.2f}%")