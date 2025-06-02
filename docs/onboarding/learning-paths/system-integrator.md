# Learning Path: System Integrator

A comprehensive guide for developers integrating ADMF-PC with external systems, data sources, and trading infrastructure.

## Overview

As a System Integrator, you'll learn how to:
- Connect external data sources
- Integrate with broker APIs
- Deploy to production environments
- Monitor system performance
- Build custom components

## Prerequisites

- Basic Python knowledge
- Understanding of APIs and networking
- Familiarity with YAML configuration
- Basic trading concepts

## Learning Path

### Phase 1: Understanding the Architecture (Week 1)

#### Day 1-2: Core Concepts
- [ ] Read [System Architecture](../../SYSTEM_ARCHITECTURE_V5.MD)
- [ ] Study [Protocol + Composition](../../architecture/03-PROTOCOL-COMPOSITION.md)
- [ ] Understand [Event-Driven Architecture](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)

#### Day 3-4: Container System
- [ ] Learn [Container Hierarchy](../../architecture/02-CONTAINER-HIERARCHY.md)
- [ ] Practice creating isolated containers
- [ ] Understand lifecycle management

#### Day 5-7: Integration Points
- [ ] Study component protocols
- [ ] Learn event bus patterns
- [ ] Understand data flow

### Phase 2: Data Integration (Week 2)

#### Custom Data Sources
```python
class MyDataProvider:
    """Example custom data provider"""
    
    def get_data(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        # Connect to your data source
        # Return standardized DataFrame
        return data
```

#### Tasks:
- [ ] Implement CSV data loader
- [ ] Create database connector
- [ ] Build API data fetcher
- [ ] Add real-time streaming

#### Example: Database Integration
```python
class PostgresDataProvider:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
    
    def get_data(self, symbol: str, start: datetime, end: datetime):
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = %s AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp
        """
        return pd.read_sql(query, self.conn, params=[symbol, start, end])
```

### Phase 3: Broker Integration (Week 3)

#### Live Trading Connection
```python
class AlpacaBroker:
    """Example broker integration"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api = alpaca.REST(api_key, secret_key)
    
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # Convert ADMF-PC order to broker format
        broker_order = self.api.submit_order(
            symbol=order['symbol'],
            qty=order['quantity'],
            side=order['action'].lower(),
            type='market',
            time_in_force='day'
        )
        return self._convert_to_fill(broker_order)
```

#### Tasks:
- [ ] Implement order translation
- [ ] Handle order status updates
- [ ] Manage positions
- [ ] Track account balance

### Phase 4: Production Deployment (Week 4)

#### Deployment Architecture
```yaml
# production_config.yaml
deployment:
  mode: distributed
  
  components:
    data_service:
      replicas: 3
      resources:
        memory: 4GB
        cpu: 2
    
    strategy_workers:
      replicas: 10
      resources:
        memory: 2GB
        cpu: 1
    
    execution_service:
      replicas: 2
      resources:
        memory: 8GB
        cpu: 4
```

#### Monitoring Setup
```python
class SystemMonitor:
    """Production monitoring"""
    
    def __init__(self, metrics_backend: str):
        self.metrics = MetricsClient(metrics_backend)
    
    def track_performance(self, event: Event):
        self.metrics.increment('events.processed')
        self.metrics.timing('event.latency', event.latency)
        
        if event.type == 'ORDER':
            self.metrics.increment('orders.created')
        elif event.type == 'FILL':
            self.metrics.increment('orders.filled')
```

### Phase 5: Advanced Integration (Ongoing)

#### Custom Component Development

1. **Protocol Implementation**
```python
class CustomIndicator:
    """Implements indicator protocol"""
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # Your custom logic
        return result
```

2. **Event Integration**
```python
class CustomEventHandler:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        event_bus.subscribe('CUSTOM_EVENT', self.handle_event)
    
    def handle_event(self, event: Event):
        # Process custom events
        pass
```

3. **Container Creation**
```python
class CustomContainer(UniversalContainer):
    """Specialized container for custom workflow"""
    
    def initialize(self, context: Dict[str, Any]):
        super().initialize(context)
        # Add custom components
        self.add_component(CustomIndicator())
        self.add_component(CustomEventHandler(self.event_bus))
```

## Integration Patterns

### 1. Adapter Pattern
```python
class ExternalSystemAdapter:
    """Adapt external system to ADMF-PC protocols"""
    
    def __init__(self, external_system):
        self.external = external_system
    
    def generate_signal(self, data):
        # Translate external format to ADMF-PC format
        external_result = self.external.analyze(data)
        return {
            "action": self._map_action(external_result.signal),
            "strength": external_result.confidence,
            "metadata": {"source": "external_system"}
        }
```

### 2. Bridge Pattern
```python
class DataBridge:
    """Bridge between data formats"""
    
    def __init__(self, source_format: str, target_format: str = "admf"):
        self.source = source_format
        self.target = target_format
    
    def convert(self, data: Any) -> pd.DataFrame:
        converters = {
            'csv': self._from_csv,
            'json': self._from_json,
            'msgpack': self._from_msgpack,
            'protobuf': self._from_protobuf
        }
        return converters[self.source](data)
```

### 3. Gateway Pattern
```python
class TradingGateway:
    """Unified interface to multiple brokers"""
    
    def __init__(self):
        self.brokers = {
            'alpaca': AlpacaBroker(),
            'ib': InteractiveBrokersBroker(),
            'binance': BinanceBroker()
        }
    
    def route_order(self, order: Dict[str, Any]):
        broker = self.brokers[order['broker']]
        return broker.execute_order(order)
```

## Best Practices

### 1. Error Handling
```python
class ResilientIntegration:
    def __init__(self):
        self.retry_policy = RetryPolicy(max_attempts=3, backoff=2.0)
    
    @with_retry
    def fetch_data(self, symbol: str):
        try:
            return self._fetch_internal(symbol)
        except NetworkError as e:
            self.logger.warning(f"Network error: {e}, retrying...")
            raise
        except DataError as e:
            self.logger.error(f"Data error: {e}, not retrying")
            return self._get_fallback_data(symbol)
```

### 2. Performance Optimization
```python
class OptimizedDataProvider:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
        self.pool = ThreadPoolExecutor(max_workers=10)
    
    def get_bulk_data(self, symbols: List[str]):
        # Parallel fetching
        futures = [
            self.pool.submit(self._get_cached_data, symbol)
            for symbol in symbols
        ]
        return {
            symbol: future.result()
            for symbol, future in zip(symbols, futures)
        }
```

### 3. Security
```python
class SecureIntegration:
    def __init__(self):
        self.credentials = self._load_encrypted_credentials()
        self.rate_limiter = RateLimiter(calls=100, period=60)
    
    @rate_limited
    def make_api_call(self, endpoint: str):
        headers = self._get_auth_headers()
        response = requests.get(endpoint, headers=headers)
        return self._validate_response(response)
```

## Common Integration Scenarios

### 1. Multi-Exchange Arbitrage
```yaml
data:
  providers:
    - type: binance_api
      symbols: ["BTC/USDT", "ETH/USDT"]
    - type: coinbase_api
      symbols: ["BTC-USD", "ETH-USD"]
    - type: kraken_api
      symbols: ["XBTUSD", "ETHUSD"]

execution:
  routers:
    - type: smart_order_router
      exchanges: ["binance", "coinbase", "kraken"]
```

### 2. Alternative Data Integration
```python
class SentimentDataProvider:
    """Integrate social media sentiment"""
    
    def get_sentiment(self, symbol: str, timestamp: datetime):
        tweets = self.twitter_api.search(f"${symbol}", since=timestamp)
        return self.sentiment_analyzer.analyze_bulk(tweets)
```

### 3. Risk System Integration
```python
class ExternalRiskSystem:
    """Connect to external risk management"""
    
    def check_compliance(self, order: Dict[str, Any]):
        response = self.risk_api.validate_order(order)
        if not response.approved:
            raise RiskViolation(response.reason)
        return True
```

## Testing Integration

### Unit Testing
```python
def test_data_provider():
    provider = MockDataProvider()
    data = provider.get_data("TEST", "2024-01-01", "2024-01-31")
    assert len(data) > 0
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close'])
```

### Integration Testing
```python
class TestBrokerIntegration:
    def test_order_lifecycle(self):
        broker = TestBroker()  # Paper trading account
        order = {"symbol": "AAPL", "quantity": 100, "action": "BUY"}
        
        fill = broker.execute_order(order)
        assert fill['status'] == 'filled'
        assert fill['filled_quantity'] == 100
```

## Resources

- [Component Protocols](../../COMPONENT_CATALOG.md)
- [Event Reference](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
- [Testing Standards](../../standards/TESTING-STANDARDS.md)
- [Production Checklist](../../complexity-guide/06-going-beyond/step-18-production-simulation.md)

## Next Steps

After completing this learning path:
1. Build a complete integration with your data source
2. Deploy a paper trading system
3. Contribute your integrations back to the community
4. Explore advanced patterns in the Complexity Guide

---

*Remember: Good integration is invisible - it just works. Focus on reliability, performance, and maintainability.*