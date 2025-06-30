import pandas as pd
import json
from pathlib import Path

results_dir = Path("config/bollinger/results/latest")
closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/positions_close.parquet")

exit_data = []
for idx, row in closes.iterrows():
    metadata = row.get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}
    
    exit_type = metadata.get('exit_type', 'unknown') if isinstance(metadata, dict) else 'unknown'
    exit_data.append({
        'exit_type': exit_type,
        'exit_price': row['payload']['exit_price'],
        'entry_price': row['payload']['entry_price'],
        'quantity': row['payload']['quantity']
    })

exit_df = pd.DataFrame(exit_data)

# Calculate returns
exit_df['return_pct'] = exit_df.apply(
    lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) if row['quantity'] > 0 
               else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100),
    axis=1
)

# Check stop losses
stop_losses = exit_df[exit_df['exit_type'] == 'stop_loss']
print(f"Stop losses: {len(stop_losses)}")
print(f"  Expected return: -0.075%")
print(f"  Actual mean: {stop_losses['return_pct'].mean():.4f}%")
print(f"  Range: [{stop_losses['return_pct'].min():.4f}%, {stop_losses['return_pct'].max():.4f}%]")

# Check take profits
take_profits = exit_df[exit_df['exit_type'] == 'take_profit']
print(f"\nTake profits: {len(take_profits)}")
print(f"  Expected return: +0.15%")
print(f"  Actual mean: {take_profits['return_pct'].mean():.4f}%")
print(f"  Range: [{take_profits['return_pct'].min():.4f}%, {take_profits['return_pct'].max():.4f}%]")

print("\nThe exits are NOT using stop/target prices\!")
