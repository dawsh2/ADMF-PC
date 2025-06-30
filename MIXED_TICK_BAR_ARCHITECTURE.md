# Mixed Tick/Bar Architecture for ADMF-PC

## Your Use Case: Bars for Strategy, Ticks for Risk

This is a great architecture! You can have:
- **Strategies** trained on bar data continuing to use bars
- **Risk management** using tick data for precise stop-loss and take-profit execution

## Architecture Design

### Option 1: Dual Data Containers (Recommended)

Create two data containers in your topology:
1. Bar data container for strategies
2. Tick data container for risk management

```yaml
# config/patterns/topologies/universal_mixed.yaml
containers:
  # Bar data for strategies
  - name_template: "{symbol}_{timeframe}_data"
    type: data
    components: ["data_streamer"]
    config:
      symbol: "{symbol}"
      timeframe: "{timeframe}"  # e.g., "1m"
      
  # Tick data for risk management
  - name_template: "{symbol}_tick_data"
    type: data
    components: ["data_streamer"]
    config:
      symbol: "{symbol}"
      timeframe: "tick"
      
  # Strategy container subscribes to BAR events from bar data
  - name: strategy
    type: strategy
    event_subscriptions:
      - event_type: BAR
        source: "{symbol}_{timeframe}_data"
        
  # Risk container subscribes to BAR events from tick data
  - name: risk
    type: risk
    event_subscriptions:
      - event_type: BAR  # Ticks are converted to bars
        source: "{symbol}_tick_data"
```

### Option 2: Single Container with Filtered Subscriptions

Modify the risk manager to subscribe to tick data:

```python
# In risk_manager.py
class RiskManager:
    def __init__(self, config):
        self.stop_loss_pct = config.get('stop_loss', 0.02)
        self.take_profit_pct = config.get('take_profit', 0.05)
        self.use_tick_data = config.get('use_tick_data', True)
        
    def setup_subscriptions(self, event_bus):
        if self.use_tick_data:
            # Subscribe to tick data for precise exits
            event_bus.subscribe(
                EventType.BAR,
                self.on_tick,
                filter_func=lambda e: e.source_id.endswith('_tick_data')
            )
        
    def on_tick(self, event):
        """Process tick data for stop-loss/take-profit."""
        tick_price = event.payload['bar'].close  # Tick converted to bar
        
        for position in self.positions.values():
            # Check stop loss
            if position.is_long:
                if tick_price <= position.stop_loss_price:
                    self.trigger_stop_loss(position, tick_price)
            else:
                if tick_price >= position.stop_loss_price:
                    self.trigger_stop_loss(position, tick_price)
                    
            # Check take profit
            if position.is_long:
                if tick_price >= position.take_profit_price:
                    self.trigger_take_profit(position, tick_price)
```

### Option 3: Hybrid Subscription Pattern

Create a component that subscribes to both:

```python
class HybridRiskManager:
    def __init__(self):
        self.latest_bar = None
        self.positions = {}
        
    def setup_subscriptions(self, event_bus):
        # Subscribe to bars for position tracking
        event_bus.subscribe(
            EventType.BAR,
            self.on_bar,
            filter_func=lambda e: not e.source_id.endswith('_tick')
        )
        
        # Subscribe to ticks for precise exits
        event_bus.subscribe(
            EventType.BAR,
            self.on_tick,
            filter_func=lambda e: e.source_id.endswith('_tick')
        )
    
    def on_bar(self, event):
        """Update position tracking with bar data."""
        self.latest_bar = event.payload['bar']
        self.update_position_metrics()
    
    def on_tick(self, event):
        """Check stops/targets with tick precision."""
        tick = event.payload['bar']
        self.check_exit_conditions(tick.close)
```

## Implementation Example

Here's how to set up mixed streaming:

### 1. Config File
```yaml
# config/mixed_streaming.yaml
name: mixed_tick_bar
description: Bars for strategy, ticks for risk

# Define both data sources
data_sources:
  bars: SPY_1m
  ticks: SPY_tick

strategies:
  - type: bollinger_bands
    data_source: bars  # Use bar data
    params:
      period: 20
      std_dev: 2.0

risk:
  data_source: ticks  # Use tick data
  stop_loss: 0.01     # 1% stop loss
  take_profit: 0.02   # 2% take profit
  use_tick_precision: true
```

### 2. Modified Main.py Support
```python
# In main.py --alpaca handling
if args.alpaca:
    # Support multiple data sources
    if 'data_sources' in config_dict:
        # Mixed mode - create multiple data configs
        for name, data_str in config_dict['data_sources'].items():
            symbol, timeframe = data_str.split('_')
            # Create data config for each source
    else:
        # Single data source (current behavior)
```

## Benefits of This Architecture

1. **Best of Both Worlds**
   - Strategies use stable bar data they were trained on
   - Risk management gets tick-level precision

2. **Reduced Latency for Exits**
   - Stop losses trigger immediately when price touches level
   - No waiting for bar close

3. **Accurate Fill Prices**
   - Exit at actual tick price, not bar close
   - Better slippage modeling

4. **Flexible Configuration**
   - Can enable/disable tick precision per strategy
   - Different timeframes for different components

## Performance Considerations

1. **Data Volume**
   - Ticks are much higher volume than bars
   - Filter ticks to only symbols with positions

2. **Processing Efficiency**
   - Risk checks should be lightweight
   - Avoid heavy calculations on every tick

3. **Network Bandwidth**
   - Each data stream uses a WebSocket connection
   - Alpaca limits concurrent connections

## Example Output

When running with mixed data:
```
Strategy using bars:
📊 Received bar: SPY 1m O:612.50 H:612.55 L:612.48 C:612.52 V:50000

Risk manager using ticks:
📊 Received tick: SPY @ $612.51 Size:100
📊 Received tick: SPY @ $612.49 Size:50  <- Stop triggered here!
🛑 Stop loss triggered at $612.49 (tick precision)
```

## Future Enhancements

1. **Tick Aggregation**
   - Volume bars (every 1000 shares)
   - Dollar bars (every $100k traded)
   - Range bars (every $1 price movement)

2. **Smart Order Routing**
   - Use tick data to optimize order placement
   - Detect favorable liquidity conditions

3. **Market Microstructure Analysis**
   - Analyze bid-ask spread from tick data
   - Detect unusual trading activity

This architecture gives you the precision of tick data where it matters most (exits) while maintaining the stability of bar data for signal generation!