#!/usr/bin/env python3

# Current results with 1 share per trade
trades = 406
wins = 188
losses = 218
avg_win = 0.87  # ~0.15% of $600
avg_loss = -0.43  # ~0.075% of $600

total_pnl = (wins * avg_win) + (losses * avg_loss)
print(f'Total P&L from 1 share per trade: ${total_pnl:.2f}')

# If notebook shows 10.27% returns on $100k
# Then total P&L should be $10,270
notebook_pnl = 100000 * 0.1027
print(f'Notebook P&L for 10.27% returns: ${notebook_pnl:.2f}')

# How many shares per trade would we need?
shares_needed = notebook_pnl / total_pnl
print(f'\nShares per trade needed to match notebook: {shares_needed:.0f}')

# Or as % of capital
capital = 100000
spy_price = 600
position_value = shares_needed * spy_price
print(f'Position value: ${position_value:,.0f}')
print(f'As % of capital: {(position_value / capital) * 100:.1f}%')

print('\nThis suggests the notebook might be using:')
print('- Fixed % of capital per trade')
print('- Multiple shares per trade')
print('- Compounding returns')
print('- Different position sizing logic')

# Additional analysis
print('\n=== Further Analysis ===')
pnl_per_trade = total_pnl / trades
return_per_trade = (pnl_per_trade / spy_price) * 100
print(f'Average P&L per trade: ${pnl_per_trade:.4f}')
print(f'Average return per trade: {return_per_trade:.4f}%')

# With 10% of capital
shares_10pct = (capital * 0.10) / spy_price
total_return_10pct = (pnl_per_trade * shares_10pct * trades / capital) * 100
print(f'\nWith 10% of capital per trade ({shares_10pct:.0f} shares):')
print(f'Total return would be: {total_return_10pct:.2f}%')