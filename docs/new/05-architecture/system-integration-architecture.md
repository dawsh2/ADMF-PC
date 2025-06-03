# System Integration Architecture

Practical guide for integrating ADMF-PC with external data sources and extending the existing CSV-based backtesting framework.

## 🎯 Overview

ADMF-PC is currently a **CSV-based backtesting and strategy research framework**. While the architecture supports extensibility, the existing integrations are limited to local file-based data processing and simulated execution. This document covers the actual integration points and realistic extension patterns for the current system.

## 🏗️ Current Integration Architecture

### What Actually Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT ADMF-PC SYSTEM                            │
│                                                                             │
│  Data Sources                 Core Processing              Output           │
│                                                                             │
│  ┌─────────────┐              ┌─────────────┐             ┌─────────────┐   │
│  │   CSV       │              │   Data      │             │  Backtest   │   │
│  │   Files     │─────────────▶│ Containers  │────────────▶│  Results    │   │
│  │  (OHLCV)    │              │             │             │             │   │
│  └─────────────┘              └─────────────┘             └─────────────┘   │
│                                       │                                     │
│                                       ▼                                     │
│                               ┌─────────────┐             ┌─────────────┐   │
│                               │  Strategy   │             │   Logs &    │   │
│                               │ Containers  │────────────▶│  Metrics    │   │
│                               │ • Momentum  │             │             │   │
│                               │ • MeanRev   │             │             │   │
│                               └─────────────┘             └─────────────┘   │
│                                       │                                     │
│                                       ▼                                     │
│                               ┌─────────────┐             ┌─────────────┐   │
│                               │  Execution  │             │ Performance │   │
│                               │   Engine    │────────────▶│  Reports    │   │
│                               │ (Simulated) │             │             │   │
│                               └─────────────┘             └─────────────┘   │
│                                                                             │
│  Working Features:                                                          │
│  • CSV data loading (SPY_1m.csv format)                                    │
│  • Protocol-based container architecture                                   │
│  • Event-driven communication between containers                           │
│  • YAML configuration system                                               │
│  • Simulated order execution and portfolio tracking                        │
│  • Basic technical indicators (RSI, MA)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Source Integration

### Current CSV Data Loading System

ADMF-PC currently supports local CSV file processing through the `SimpleCSVLoader` class:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CSV DATA LOADING ARCHITECTURE                        │
│                                                                             │
│  File Formats              Data Processing             Normalized Output    │
│                                                                             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │   CSV       │           │   Column    │           │ Standard    │       │
│  │  Files      │──────────▶│ Mapping &   │──────────▶│ DataFrame   │       │
│  │             │           │ Validation  │           │ (OHLCV)     │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│                                    │                                        │
│  Supported Formats:                │                                        │
│  • symbol.csv                      ▼                                        │
│  • SYMBOL_1m.csv              ┌─────────────┐                              │
│  • SYMBOL_daily.csv           │   Missing   │                              │
│  • Multiple delimiters        │    Data     │                              │
│  • Various date formats       │  Handling   │                              │
│                                └─────────────┘                              │
│                                                                             │
│  Data Validation:                                                           │
│  • OHLC relationship validation (High ≥ Open,Close,Low)                     │
│  • Volume ≥ 0 validation                                                    │
│  • Timestamp parsing and sorting                                           │
│  • Forward fill for missing prices, zero fill for volume                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### YAML Configuration for Data Loading

```yaml
# config/spy_momentum_backtest.yaml (Working Example)
name: SPY Momentum Strategy Backtest
workflow_type: backtest

data:
  source: csv                    # Only working source
  file_path: data/SPY_1m.csv    # Must exist locally
  symbols: ["SPY"]              # Must match filename
  timeframe: "1m"               # For reference only

strategies:
  - name: momentum_strategy     # Implemented strategy
    type: momentum
    parameters:
      lookback_period: 20
      momentum_threshold: 0.0002
      rsi_period: 14

portfolio:
  initial_capital: 100000      # Simulated starting capital

backtest:                      # Simulation settings
  commission: 0.0    # Alpaca has zero commission
  slippage: 0.0005   # 0.05% for liquid stocks
```

### Data Extension Patterns

To add new data sources, implement the DataLoader protocol:

```python
# Extension pattern for new data sources
class CustomDataLoader:
    """Custom data loader following ADMF-PC protocols"""
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data from custom source"""
        # Your data loading logic here
        # Must return DataFrame with OHLCV columns
        # Index must be datetime
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate data meets ADMF-PC requirements"""
        required = ["open", "high", "low", "close", "volume"]
        return all(col in df.columns for col in required)

# Register in factory
def create_data_loader(loader_type: str, **config):
    loaders = {
        'csv': SimpleCSVLoader,
        'custom': CustomDataLoader,  # Add your loader
        # Add other loaders as needed
    }
    return loaders[loader_type](**config)
```

## 💹 Execution Integration

### Current Backtest-Only System

ADMF-PC currently provides simulated execution through the `BacktestBrokerRefactored` class:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SIMULATED EXECUTION ARCHITECTURE                       │
│                                                                             │
│  Strategy Signals            Order Processing           Portfolio State     │
│                                                                             │
│  ┌─────────────┐             ┌─────────────┐           ┌─────────────┐     │
│  │   Signal    │             │   Order     │           │ Portfolio   │     │
│  │ Generator   │────────────▶│ Validation  │──────────▶│   State     │     │
│  │             │             │             │           │ Tracking    │     │
│  └─────────────┘             └─────────────┘           └─────────────┘     │
│                                      │                                      │
│                                      ▼                                      │
│                              ┌─────────────┐           ┌─────────────┐     │
│                              │   Market    │           │   Fill      │     │
│                              │ Simulator   │──────────▶│  Events     │     │
│                              │             │           │             │     │
│                              └─────────────┘           └─────────────┘     │
│                                                                             │
│  Features:                                                                  │
│  • Order validation against available cash/positions                       │
│  • Simulated slippage and commission costs                                 │
│  • Market and limit order types                                            │
│  • Position tracking and P&L calculation                                   │
│  • No real broker connections                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Order Lifecycle in Current System

```python
# From backtest_broker_refactored.py - Actual implementation
class BacktestBrokerRefactored:
    """Simulated broker for backtesting only"""
    
    def submit_order(self, order: Order) -> str:
        """Submit simulated order"""
        # Validate against portfolio state
        if not self._validate_order(order):
            return order.order_id  # Rejected
            
        # Store order for tracking
        self.order_tracker.orders[order.order_id] = order
        
        # Simulate immediate fill (no market delay)
        fill = self._create_simulated_fill(order)
        self._process_fill(fill)
        
        return order.order_id
```

### Market Simulation Parameters

ADMF-PC includes configurable market simulation for realistic backtesting:

```yaml
# Realistic Alpaca parameters for liquid stocks
market_simulation:
  # Alpaca has zero commission
  commission_model:
    type: "zero"  # No commission for Alpaca
    
  # Slippage for liquid stocks (non-DMA execution)
  slippage_model:
    type: "percentage"
    base_slippage_pct: 0.0005  # 0.05% for liquid stocks
    # Can increase to 0.0015 (0.15%) for less liquid
    
  # High fill probability for market orders
  fill_probability: 0.98  # 98% for liquid stocks
  
  # Partial fills possible for large orders
  partial_fill_enabled: true
  max_volume_participation: 0.1  # Max 10% of volume
```

#### Market Simulation Models

```python
# Available in market_simulation.py
class FixedSlippageModel:
    """Fixed percentage slippage"""
    slippage_percent: float = 0.001  # 0.1% default
    
class VolumeSlippageModel:
    """Volume-based slippage for size impact"""
    base_impact: float = 0.0001
    # Impact increases with sqrt(order_ratio)
    
class ZeroCommissionModel:
    """For Alpaca and other zero-commission brokers"""
    commission_per_trade: float = 0.0
```

### Extension Patterns for Live Trading

To add live trading capability, implement the Broker protocol:

```python
# Extension pattern for Alpaca broker integration
class AlpacaBrokerAdapter:
    """Adapter for Alpaca API integration"""
    
    def __init__(self, alpaca_api):
        self.api = alpaca_api  # Alpaca-py client
        
    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca"""
        # Convert ADMF-PC order to Alpaca format
        alpaca_order = self._convert_order(order)
        
        # Submit to Alpaca API
        response = self.api.submit_order(**alpaca_order)
        
        # Return order ID for tracking
        return response.id
        
    def _convert_order(self, order: Order) -> dict:
        """Convert ADMF-PC order to Alpaca format"""
        return {
            'symbol': order.symbol,
            'qty': order.quantity,
            'side': order.side.value.lower(),
            'type': order.order_type.value.lower(),
            'time_in_force': 'day',  # Alpaca default
            # No commission parameters needed - Alpaca is zero commission
        }
        
    def get_realistic_slippage(self, symbol: str) -> float:
        """Get realistic slippage based on liquidity"""
        # Could query Alpaca for bid-ask spread
        if symbol in ['SPY', 'QQQ', 'AAPL']:  # Highly liquid
            return 0.0005  # 0.05%
        else:
            return 0.0015  # 0.15% for less liquid
```

## 🛡️ Risk System Integration

### Current Risk Management

ADMF-PC includes basic risk management through the `PortfolioState` class:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT RISK MANAGEMENT                              │
│                                                                             │
│  Position Tracking           Risk Checks               Portfolio State      │
│                                                                             │
│  ┌─────────────┐            ┌─────────────┐          ┌─────────────┐       │
│  │   Order     │            │  Position   │          │   Cash      │       │
│  │ Validation  │───────────▶│   Limits    │─────────▶│  Balance    │       │
│  │             │            │             │          │             │       │
│  └─────────────┘            └─────────────┘          └─────────────┘       │
│                                    │                                        │
│                                    ▼                                        │
│                            ┌─────────────┐          ┌─────────────┐       │
│                            │  Exposure   │          │ Position    │       │
│                            │   Limits    │─────────▶│  Tracking   │       │
│                            │             │          │             │       │
│                            └─────────────┘          └─────────────┘       │
│                                                                             │
│  Current Features:                                                          │
│  • Basic position size validation                                          │
│  • Cash balance tracking                                                   │
│  • Simple percentage-based position sizing                                 │
│  • P&L calculation                                                         │
│  • No sophisticated risk models                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Monitoring and Logging

### Current Logging System

ADMF-PC includes structured logging through Python's logging module:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT LOGGING SYSTEM                            │
│                                                                             │
│  Event Sources              Log Processing             Output               │
│                                                                             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │ Container   │           │  Structured │           │   Console   │       │
│  │  Events     │──────────▶│   Logger    │──────────▶│   Output    │       │
│  │             │           │             │           │             │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│                                    │                                        │
│  ┌─────────────┐                   │                 ┌─────────────┐       │
│  │ Strategy    │                   ▼                 │  Debug Log  │       │
│  │ Decisions   │           ┌─────────────┐           │    File     │       │
│  │             │           │    Event    │──────────▶│             │       │
│  └─────────────┘           │  Tracking   │           └─────────────┘       │
│                            └─────────────┘                                 │
│  ┌─────────────┐                                                           │
│  │   Order     │                                                           │
│  │   Flow      │                                                           │
│  │             │                                                           │
│  └─────────────┘                                                           │
│                                                                             │
│  Current Capabilities:                                                      │
│  • Component-level logging with correlation IDs                            │
│  • Event tracking through container lifecycle                              │
│  • Debug logging to debug.log file                                         │
│  • Console output with configurable verbosity                              │
│  • No external monitoring system integration                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Extension Patterns

### Adding New Components

ADMF-PC's protocol-based architecture makes adding new components straightforward:

```python
# Pattern for adding new strategy types
class CustomStrategy:
    """Custom strategy following ADMF-PC protocols"""
    
    def __init__(self, **params):
        self.params = params
        
    def generate_signal(self, data: pd.DataFrame) -> dict:
        """Generate trading signal from market data"""
        # Your strategy logic here
        return {
            "action": "BUY|SELL|HOLD",
            "strength": 0.75,  # 0.0 to 1.0
            "confidence": 0.85,
            "metadata": {"reason": "your logic"}
        }

# Register in factory
strategy_factory = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'custom': CustomStrategy,  # Add your strategy
}
```

### Adding New Indicators

```python
# Pattern for custom technical indicators
class CustomIndicator:
    """Custom indicator following ADMF-PC protocols"""
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values"""
        # Your indicator calculation
        return indicator_values
        
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        return len(data) >= self.min_periods
```

## 🚀 Realistic Integration Roadmap

### Phase 1: Enhanced Data Sources (Achievable)
- Database connection (PostgreSQL, MySQL)
- API data fetchers (Alpha Vantage, Yahoo Finance)
- Real-time data streaming (WebSocket connections)
- Alternative data sources (news, sentiment)

### Phase 2: Paper Trading (Medium Complexity)
- Broker API adapters (Alpaca, TD Ameritrade)
- Real-time order submission and tracking
- Live portfolio monitoring
- Paper trading simulation

### Phase 3: Production Trading (High Complexity)
- Live trading with real money
- Advanced risk management
- Regulatory compliance
- Professional monitoring and alerting

## 📝 Key Takeaways

### What ADMF-PC Actually Is

1. **CSV-Based Backtesting Framework**: Excellent for strategy research and historical analysis
2. **Protocol-Based Architecture**: Clean, extensible design that supports new components
3. **Zero-Code Configuration**: Strategy development through YAML without programming
4. **Event-Driven System**: Clean separation between data, strategy, and execution layers

### What ADMF-PC Is Not (Currently)

1. **Live Trading System**: No real broker connections or live trading capability
2. **Real-Time Data Platform**: No live market data feeds
3. **Production-Ready**: Limited to research and backtesting environments
4. **ML/AI Platform**: Basic technical analysis only, no machine learning integration

### Extension Strategy

The most practical approach for extending ADMF-PC is to:
1. Start with enhanced data sources (databases, APIs)
2. Add paper trading capabilities using existing broker APIs
3. Gradually build toward live trading with proper risk controls
4. Maintain the zero-code YAML configuration approach

The existing architecture provides excellent foundations for these extensions while maintaining the system's core principles of simplicity and composability.

---

Continue to [Production Deployment Architecture](production-deployment-architecture.md) for scaling and operational considerations →