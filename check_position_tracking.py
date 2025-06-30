#!/usr/bin/env python3
"""
Check how positions are tracked in the traces
"""
import pandas as pd
import glob
import os

def check_position_tracking(workspace_path):
    """Check position quantities in trace data"""
    
    # Load traces
    pattern = os.path.join(workspace_path, "traces", "*", "trace_*.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No trace files found in {workspace_path}")
        return
        
    print(f"Found {len(files)} trace files")
    
    # Load first file to check structure
    df = pd.read_parquet(files[0])
    print(f"\nColumns in trace: {df.columns.tolist()}")
    
    # Look for position events
    position_events = df[df['event_type'].isin(['POSITION_OPENED', 'POSITION_CLOSED', 'ORDER_FILLED'])].copy()
    
    if len(position_events) > 0:
        print(f"\nFound {len(position_events)} position-related events")
        print("\nFirst few events:")
        
        for idx, event in position_events.head(10).iterrows():
            print(f"\nEvent: {event['event_type']}")
            print(f"  Timestamp: {event['timestamp']}")
            print(f"  Price: {event.get('price', 'N/A')}")
            
            # Check metadata
            metadata = event.get('metadata', {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
                    
            if metadata:
                print(f"  Metadata: {metadata}")
                
            # Look for quantity/size fields
            for field in ['quantity', 'size', 'position_size', 'fill_quantity']:
                if field in event:
                    print(f"  {field}: {event[field]}")
                if field in metadata:
                    print(f"  metadata.{field}: {metadata[field]}")
                    
            # Check for side/direction
            for field in ['side', 'direction', 'order_side']:
                if field in event:
                    print(f"  {field}: {event[field]}")
                if field in metadata:
                    print(f"  metadata.{field}: {metadata[field]}")
    
    # Look for the actual position tracking
    print("\n" + "="*60)
    print("Checking fill processing...")
    
    fills = df[df['event_type'] == 'ORDER_FILLED']
    if len(fills) > 0:
        print(f"\nFound {len(fills)} fills")
        
        # Group by strategy to track position evolution
        for strategy_id in fills['strategy_id'].unique()[:3]:  # First 3 strategies
            print(f"\n\nStrategy: {strategy_id}")
            strategy_fills = fills[fills['strategy_id'] == strategy_id].sort_values('timestamp')
            
            position = 0
            for idx, fill in strategy_fills.head(10).iterrows():
                metadata = fill.get('metadata', {})
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                
                # Get fill details
                side = metadata.get('side', 'UNKNOWN')
                quantity = metadata.get('quantity', fill.get('quantity', 0))
                
                # Calculate position change
                if side == 'BUY':
                    position += quantity
                elif side == 'SELL':
                    position -= quantity
                    
                print(f"\n  Fill at {fill['timestamp']}")
                print(f"    Side: {side}, Quantity: {quantity}")
                print(f"    Position after: {position}")
                
                # Check if this creates a short position
                if position < 0:
                    print(f"    ⚠️  SHORT POSITION: {position}")

# Usage
if __name__ == "__main__":
    # Replace with your workspace path
    workspace = "results/latest"
    
    print("Checking position tracking in traces...")
    print("This will show how positions evolve with each fill")
    print("We should see negative positions for shorts")
    print()
    
    # You can run this with:
    # check_position_tracking("path/to/your/workspace")