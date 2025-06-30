import pandas as pd

# Read positions close
positions_close = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_close/positions_close.parquet')
print(f'=== POSITIONS CLOSE ===')
print(f'Total positions closed: {len(positions_close)}')

# Show first few rows
print('\nFirst 5 rows:')
for i in range(min(5, len(positions_close))):
    row = positions_close.iloc[i]
    metadata = row['metadata']
    if isinstance(metadata, dict):
        # Extract key fields
        qty = metadata.get('quantity', 0)
        pnl = metadata.get('realized_pnl', 0)
        entry_price = metadata.get('entry_price', 0)
        exit_price = metadata.get('exit_price', 0)
        exit_type = metadata.get('metadata', {}).get('exit_type', 'N/A') if 'metadata' in metadata else 'N/A'
        
        print(f'  {i+1}. qty={qty:.1f}, pnl=${pnl:.2f}, entry=${entry_price:.2f}, exit=${exit_price:.2f}, exit_type={exit_type}')

# Look for shorts with stop losses
print('\nShorts with stop loss exits (first 10):')
count = 0
for i in range(len(positions_close)):
    row = positions_close.iloc[i]
    metadata = row['metadata']
    if isinstance(metadata, dict):
        qty = metadata.get('quantity', 0)
        pnl = metadata.get('realized_pnl', 0)
        exit_type = metadata.get('metadata', {}).get('exit_type', '') if 'metadata' in metadata else ''
        
        if qty < 0 and exit_type == 'stop_loss':
            entry_price = metadata.get('entry_price', 0)
            exit_price = metadata.get('exit_price', 0)
            # For shorts: profit when exit < entry, loss when exit > entry
            expected_pnl_sign = 'profit' if exit_price < entry_price else 'loss'
            actual_pnl_sign = 'profit' if pnl > 0 else 'loss'
            
            print(f'  qty={qty:.1f}, entry=${entry_price:.2f}, exit=${exit_price:.2f}, pnl=${pnl:.2f} ({actual_pnl_sign}), expected={expected_pnl_sign}')
            count += 1
            if count >= 10:
                break

# Calculate overall metrics
total_pnl = 0
long_pnl = 0
short_pnl = 0
long_trades = 0
short_trades = 0
wins = 0
losses = 0

for i in range(len(positions_close)):
    metadata = positions_close.iloc[i]['metadata']
    if isinstance(metadata, dict):
        qty = metadata.get('quantity', 0)
        pnl = metadata.get('realized_pnl', 0)
        total_pnl += pnl
        
        if qty > 0:
            long_pnl += pnl
            long_trades += 1
        else:
            short_pnl += pnl
            short_trades += 1
            
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

print(f'\n=== P&L SUMMARY ===')
print(f'Total trades: {len(positions_close)}')
print(f'Long trades: {long_trades}')
print(f'Short trades: {short_trades}')
print(f'Wins: {wins}')
print(f'Losses: {losses}')
print(f'Win rate: {wins/len(positions_close)*100:.1f}%' if len(positions_close) > 0 else 'N/A')
print(f'\nTotal P&L: ${total_pnl:.2f}')
print(f'Long P&L: ${long_pnl:.2f}')
print(f'Short P&L: ${short_pnl:.2f}')

# Assuming initial capital of 100000
initial_capital = 100000
returns = (initial_capital + total_pnl) / initial_capital - 1
print(f'\nReturns: {returns*100:.2f}%')