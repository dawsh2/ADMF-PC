"""Debug order flow to understand why stop/target prices aren't being used."""

import pandas as pd
import pyarrow.parquet as pq

# Load the order and fill data
fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'

try:
    orders_df = pd.read_parquet(orders_path)
    print(f"Found {len(orders_df)} orders")
    
    # Check orders with exit metadata
    exit_orders = orders_df[orders_df['metadata'].apply(lambda x: 'exit_type' in x if isinstance(x, dict) else False)]
    print(f"\nFound {len(exit_orders)} exit orders")
    
    if len(exit_orders) > 0:
        print("\nSample exit orders:")
        for idx, order in exit_orders.head(5).iterrows():
            metadata = order['metadata']
            exit_type = metadata.get('exit_type', 'unknown')
            price = order.get('price', 'N/A')
            print(f"  Order {order['order_id']}: exit_type={exit_type}, price={price}")
            print(f"    Full metadata: {metadata}")
    
except Exception as e:
    print(f"Error loading orders: {e}")

try:
    fills_df = pd.read_parquet(fills_path)
    print(f"\nFound {len(fills_df)} fills")
    
    # Check fills with exit metadata
    exit_fills = fills_df[fills_df['metadata'].apply(lambda x: 'exit_type' in x if isinstance(x, dict) else False)]
    print(f"\nFound {len(exit_fills)} exit fills")
    
    if len(exit_fills) > 0:
        # Group by exit type
        exit_type_counts = exit_fills['metadata'].apply(lambda x: x.get('exit_type', 'unknown')).value_counts()
        print("\nExit type distribution:")
        print(exit_type_counts)
        
        # Check stop loss fills
        stop_loss_fills = exit_fills[exit_fills['metadata'].apply(lambda x: x.get('exit_type') == 'stop_loss')]
        if len(stop_loss_fills) > 0:
            print(f"\nStop loss fills: {len(stop_loss_fills)}")
            print("Sample stop loss fills:")
            for idx, fill in stop_loss_fills.head(3).iterrows():
                print(f"  Fill {fill['fill_id']}: price={fill['price']}")
                print(f"    Order ID: {fill.get('order_id', 'N/A')}")
                print(f"    Metadata: {fill['metadata']}")
        
        # Check take profit fills
        take_profit_fills = exit_fills[exit_fills['metadata'].apply(lambda x: x.get('exit_type') == 'take_profit')]
        if len(take_profit_fills) > 0:
            print(f"\nTake profit fills: {len(take_profit_fills)}")
            print("Sample take profit fills:")
            for idx, fill in take_profit_fills.head(3).iterrows():
                print(f"  Fill {fill['fill_id']}: price={fill['price']}")
                print(f"    Order ID: {fill.get('order_id', 'N/A')}")
                print(f"    Metadata: {fill['metadata']}")
    
except Exception as e:
    print(f"Error loading fills: {e}")

# Check if orders have prices set
print("\n\nChecking order prices...")
try:
    if len(exit_orders) > 0:
        # Check if price field exists and is populated
        prices_set = exit_orders['price'].notna().sum()
        print(f"Exit orders with price set: {prices_set}/{len(exit_orders)}")
        
        # Show distribution of prices
        price_values = exit_orders['price'].dropna()
        if len(price_values) > 0:
            print(f"Price range: {price_values.min()} to {price_values.max()}")
            print(f"Non-zero prices: {(price_values > 0).sum()}")
            
            # Sample non-zero prices
            non_zero_prices = exit_orders[exit_orders['price'] > 0]
            if len(non_zero_prices) > 0:
                print("\nSample orders with non-zero prices:")
                for idx, order in non_zero_prices.head(3).iterrows():
                    print(f"  Order {order['order_id']}: price={order['price']}, exit_type={order['metadata'].get('exit_type')}")
except Exception as e:
    print(f"Error checking prices: {e}")