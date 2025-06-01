# Step 13: Cross-Exchange Arbitrage

**Status**: Advanced Step  
**Complexity**: Very High  
**Prerequisites**: [Step 12: Crypto & DeFi Integration](step-12-crypto-defi.md) completed  
**Architecture Ref**: [Arbitrage Architecture](../architecture/arbitrage-architecture.md)

## 🎯 Objective

Implement sophisticated cross-exchange arbitrage system:
- Multi-venue price discovery and aggregation
- Low-latency order routing and execution
- Risk management for arbitrage positions
- Transaction cost optimization
- Market impact minimization
- Inventory management across venues
- Real-time P&L tracking

## 📋 Required Reading

Before starting:
1. [High-Frequency Trading Systems](../references/hft-systems.md)
2. [Arbitrage Strategy Design](../references/arbitrage-strategies.md)
3. [Latency Optimization Techniques](../references/latency-optimization.md)
4. [Market Microstructure](../references/market-microstructure.md)

## 🏗️ Implementation Tasks

### 1. Arbitrage Engine Core

```python
# src/arbitrage/engine.py
from typing import Dict, List, Optional, Tuple, Set, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import asyncio
from collections import defaultdict
import heapq
from sortedcontainers import SortedDict

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    TRIANGULAR = "triangular"
    CROSS_EXCHANGE = "cross_exchange"
    STATISTICAL = "statistical"
    LATENCY = "latency"
    FUNDING_RATE = "funding_rate"

class ExecutionStrategy(Enum):
    """Execution strategies for arbitrage"""
    AGGRESSIVE = "aggressive"  # Take liquidity
    PASSIVE = "passive"  # Provide liquidity
    MIXED = "mixed"  # Optimize between both
    STEALTH = "stealth"  # Minimize market impact

@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    opportunity_id: str
    type: ArbitrageType
    symbol: str
    
    # Venues
    buy_venue: str
    sell_venue: str
    
    # Prices and amounts
    buy_price: Decimal
    sell_price: Decimal
    max_quantity: Decimal
    
    # Profitability
    gross_profit: Decimal
    net_profit: Decimal
    profit_percentage: Decimal
    
    # Risk metrics
    confidence_score: float
    execution_risk: float
    
    # Timing
    discovered_at: datetime
    expires_at: datetime
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArbitrageExecution:
    """Tracks arbitrage execution"""
    opportunity: ArbitrageOpportunity
    
    # Orders
    buy_orders: List[Any] = field(default_factory=list)
    sell_orders: List[Any] = field(default_factory=list)
    
    # Execution status
    status: str = "pending"
    executed_quantity: Decimal = Decimal("0")
    
    # Actual P&L
    actual_buy_cost: Decimal = Decimal("0")
    actual_sell_revenue: Decimal = Decimal("0")
    actual_fees: Decimal = Decimal("0")
    actual_profit: Decimal = Decimal("0")
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Risk metrics
    slippage: Decimal = Decimal("0")
    reversion_risk: float = 0.0

class CrossExchangeArbitrageEngine:
    """Main arbitrage engine for cross-exchange opportunities"""
    
    def __init__(self, exchanges: Dict[str, Any], config: Dict[str, Any]):
        self.exchanges = exchanges
        self.config = config
        
        # Market data
        self.order_books = defaultdict(dict)
        self.tickers = defaultdict(dict)
        self.trade_history = defaultdict(list)
        
        # Opportunity tracking
        self.opportunities = SortedDict()  # Sorted by profit
        self.active_executions = {}
        
        # Risk management
        self.position_tracker = PositionTracker()
        self.risk_manager = ArbitrageRiskManager(config['risk'])
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Execution engine
        self.execution_engine = ExecutionEngine(exchanges)
        
        # Configuration
        self.min_profit_threshold = Decimal(str(config.get('min_profit_usd', 10)))
        self.max_position_size = Decimal(str(config.get('max_position_size', 100000)))
        
        self.logger = ComponentLogger("ArbitrageEngine", "arbitrage")
    
    async def start(self):
        """Start the arbitrage engine"""
        self.logger.info("Starting cross-exchange arbitrage engine")
        
        # Start market data feeds
        await self._start_market_data_feeds()
        
        # Start main loops
        tasks = [
            self._opportunity_discovery_loop(),
            self._execution_loop(),
            self._risk_monitoring_loop(),
            self._performance_tracking_loop()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _start_market_data_feeds(self):
        """Start market data feeds from all exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            # Subscribe to order books
            symbols = self.config['symbols']
            
            for symbol in symbols:
                await exchange.subscribe_order_book(
                    symbol,
                    lambda ob, ex=exchange_name, s=symbol: self._handle_order_book_update(ex, s, ob)
                )
                
                await exchange.subscribe_ticker(
                    symbol,
                    lambda ticker, ex=exchange_name, s=symbol: self._handle_ticker_update(ex, s, ticker)
                )
    
    def _handle_order_book_update(self, exchange: str, symbol: str, order_book: Dict):
        """Handle order book updates"""
        self.order_books[symbol][exchange] = {
            'bids': order_book['bids'],
            'asks': order_book['asks'],
            'timestamp': order_book['timestamp']
        }
        
        # Trigger opportunity discovery
        asyncio.create_task(self._discover_opportunities(symbol))
    
    def _handle_ticker_update(self, exchange: str, symbol: str, ticker: Dict):
        """Handle ticker updates"""
        self.tickers[symbol][exchange] = ticker
    
    async def _discover_opportunities(self, symbol: str):
        """Discover arbitrage opportunities for a symbol"""
        if symbol not in self.order_books:
            return
        
        exchanges_with_data = list(self.order_books[symbol].keys())
        
        if len(exchanges_with_data) < 2:
            return
        
        # Check all exchange pairs
        for i, exchange1 in enumerate(exchanges_with_data):
            for exchange2 in exchanges_with_data[i+1:]:
                opportunity = self._check_arbitrage_opportunity(
                    symbol, exchange1, exchange2
                )
                
                if opportunity:
                    self._add_opportunity(opportunity)
    
    def _check_arbitrage_opportunity(self, symbol: str, 
                                    exchange1: str, exchange2: str) -> Optional[ArbitrageOpportunity]:
        """Check for arbitrage opportunity between two exchanges"""
        
        ob1 = self.order_books[symbol][exchange1]
        ob2 = self.order_books[symbol][exchange2]
        
        # Get best bid/ask from each exchange
        if not ob1['bids'] or not ob1['asks'] or not ob2['bids'] or not ob2['asks']:
            return None
        
        best_bid1 = Decimal(str(ob1['bids'][0][0]))
        best_ask1 = Decimal(str(ob1['asks'][0][0]))
        best_bid2 = Decimal(str(ob2['bids'][0][0]))
        best_ask2 = Decimal(str(ob2['asks'][0][0]))
        
        # Check for arbitrage opportunities
        opportunity = None
        
        # Buy on exchange1, sell on exchange2
        if best_ask1 < best_bid2:
            max_qty = min(
                Decimal(str(ob1['asks'][0][1])),
                Decimal(str(ob2['bids'][0][1]))
            )
            
            gross_profit = (best_bid2 - best_ask1) * max_qty
            fees = self._calculate_fees(exchange1, exchange2, symbol, max_qty)
            net_profit = gross_profit - fees
            
            if net_profit > self.min_profit_threshold:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"{symbol}_{exchange1}_{exchange2}_{datetime.now().timestamp()}",
                    type=ArbitrageType.CROSS_EXCHANGE,
                    symbol=symbol,
                    buy_venue=exchange1,
                    sell_venue=exchange2,
                    buy_price=best_ask1,
                    sell_price=best_bid2,
                    max_quantity=max_qty,
                    gross_profit=gross_profit,
                    net_profit=net_profit,
                    profit_percentage=(net_profit / (best_ask1 * max_qty)) * 100,
                    confidence_score=self._calculate_confidence(ob1, ob2),
                    execution_risk=self._calculate_execution_risk(exchange1, exchange2, max_qty),
                    discovered_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=5)
                )
        
        # Buy on exchange2, sell on exchange1
        elif best_ask2 < best_bid1:
            max_qty = min(
                Decimal(str(ob2['asks'][0][1])),
                Decimal(str(ob1['bids'][0][1]))
            )
            
            gross_profit = (best_bid1 - best_ask2) * max_qty
            fees = self._calculate_fees(exchange2, exchange1, symbol, max_qty)
            net_profit = gross_profit - fees
            
            if net_profit > self.min_profit_threshold:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"{symbol}_{exchange2}_{exchange1}_{datetime.now().timestamp()}",
                    type=ArbitrageType.CROSS_EXCHANGE,
                    symbol=symbol,
                    buy_venue=exchange2,
                    sell_venue=exchange1,
                    buy_price=best_ask2,
                    sell_price=best_bid1,
                    max_quantity=max_qty,
                    gross_profit=gross_profit,
                    net_profit=net_profit,
                    profit_percentage=(net_profit / (best_ask2 * max_qty)) * 100,
                    confidence_score=self._calculate_confidence(ob2, ob1),
                    execution_risk=self._calculate_execution_risk(exchange2, exchange1, max_qty),
                    discovered_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=5)
                )
        
        return opportunity
    
    def _calculate_fees(self, buy_exchange: str, sell_exchange: str, 
                       symbol: str, quantity: Decimal) -> Decimal:
        """Calculate total fees for arbitrage"""
        # Exchange fees
        buy_fee_rate = Decimal(str(self.config['exchanges'][buy_exchange]['taker_fee']))
        sell_fee_rate = Decimal(str(self.config['exchanges'][sell_exchange]['taker_fee']))
        
        buy_fee = quantity * self.order_books[symbol][buy_exchange]['asks'][0][0] * buy_fee_rate
        sell_fee = quantity * self.order_books[symbol][sell_exchange]['bids'][0][0] * sell_fee_rate
        
        # Network/withdrawal fees if applicable
        network_fee = Decimal("0")
        if self.config.get('include_network_fees', False):
            network_fee = Decimal(str(self.config.get('network_fee', 0)))
        
        return buy_fee + sell_fee + network_fee
    
    def _calculate_confidence(self, buy_ob: Dict, sell_ob: Dict) -> float:
        """Calculate confidence score for opportunity"""
        confidence = 1.0
        
        # Check order book depth
        if len(buy_ob['asks']) < 3 or len(sell_ob['bids']) < 3:
            confidence *= 0.7
        
        # Check spread consistency
        buy_spread = float(buy_ob['asks'][0][0]) - float(buy_ob['bids'][0][0])
        sell_spread = float(sell_ob['asks'][0][0]) - float(sell_ob['bids'][0][0])
        
        if buy_spread > float(buy_ob['asks'][0][0]) * 0.001:  # > 0.1%
            confidence *= 0.9
        if sell_spread > float(sell_ob['bids'][0][0]) * 0.001:
            confidence *= 0.9
        
        # Check data freshness
        now = datetime.now().timestamp() * 1000
        buy_age = (now - buy_ob['timestamp']) / 1000  # seconds
        sell_age = (now - sell_ob['timestamp']) / 1000
        
        if buy_age > 1 or sell_age > 1:  # Data older than 1 second
            confidence *= 0.8
        
        return confidence
    
    def _calculate_execution_risk(self, buy_exchange: str, sell_exchange: str, 
                                 quantity: Decimal) -> float:
        """Calculate execution risk"""
        risk = 0.0
        
        # Exchange reliability scores
        reliability_scores = {
            'binance': 0.95,
            'coinbase': 0.93,
            'kraken': 0.90,
            'default': 0.85
        }
        
        buy_reliability = reliability_scores.get(buy_exchange, reliability_scores['default'])
        sell_reliability = reliability_scores.get(sell_exchange, reliability_scores['default'])
        
        risk += (2 - buy_reliability - sell_reliability) * 0.3
        
        # Size risk (larger orders have higher risk)
        if quantity > Decimal("10000"):
            risk += 0.2
        elif quantity > Decimal("50000"):
            risk += 0.4
        
        # Latency risk
        avg_latency = self._get_average_latency(buy_exchange, sell_exchange)
        if avg_latency > 100:  # ms
            risk += 0.1
        if avg_latency > 200:
            risk += 0.2
        
        return min(risk, 1.0)
    
    def _add_opportunity(self, opportunity: ArbitrageOpportunity):
        """Add opportunity to tracking"""
        # Check if it passes risk filters
        if not self.risk_manager.validate_opportunity(opportunity):
            return
        
        # Add to sorted opportunities
        self.opportunities[opportunity.net_profit] = opportunity
        
        # Limit number of tracked opportunities
        if len(self.opportunities) > 100:
            # Remove least profitable
            self.opportunities.popitem(0)
    
    async def _opportunity_discovery_loop(self):
        """Main loop for opportunity discovery"""
        while True:
            try:
                # Clean expired opportunities
                now = datetime.now()
                expired_keys = []
                
                for profit, opp in self.opportunities.items():
                    if opp.expires_at < now:
                        expired_keys.append(profit)
                
                for key in expired_keys:
                    del self.opportunities[key]
                
                await asyncio.sleep(0.1)  # 100ms loop
                
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(1)
    
    async def _execution_loop(self):
        """Main loop for executing arbitrage"""
        while True:
            try:
                if not self.opportunities:
                    await asyncio.sleep(0.01)
                    continue
                
                # Get best opportunity
                best_profit, best_opportunity = self.opportunities.peekitem(-1)
                
                # Check if we should execute
                if self._should_execute(best_opportunity):
                    # Remove from opportunities
                    self.opportunities.popitem(-1)
                    
                    # Execute arbitrage
                    execution = await self._execute_arbitrage(best_opportunity)
                    
                    # Track execution
                    self.active_executions[execution.opportunity.opportunity_id] = execution
                
                await asyncio.sleep(0.001)  # 1ms loop for low latency
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(0.1)
    
    def _should_execute(self, opportunity: ArbitrageOpportunity) -> bool:
        """Determine if we should execute an opportunity"""
        
        # Check if still valid
        if opportunity.expires_at < datetime.now():
            return False
        
        # Check risk limits
        if not self.risk_manager.check_limits(opportunity):
            return False
        
        # Check position limits
        current_position = self.position_tracker.get_position(opportunity.symbol)
        if abs(current_position) + opportunity.max_quantity > self.max_position_size:
            return False
        
        # Check minimum profit
        if opportunity.net_profit < self.min_profit_threshold:
            return False
        
        # Check confidence threshold
        if opportunity.confidence_score < 0.7:
            return False
        
        return True
    
    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> ArbitrageExecution:
        """Execute arbitrage opportunity"""
        execution = ArbitrageExecution(opportunity=opportunity)
        execution.started_at = datetime.now()
        
        try:
            # Determine execution strategy
            strategy = self._determine_execution_strategy(opportunity)
            
            # Place orders based on strategy
            if strategy == ExecutionStrategy.AGGRESSIVE:
                await self._execute_aggressive(execution)
            elif strategy == ExecutionStrategy.PASSIVE:
                await self._execute_passive(execution)
            elif strategy == ExecutionStrategy.STEALTH:
                await self._execute_stealth(execution)
            else:
                await self._execute_mixed(execution)
            
            # Update execution status
            execution.status = "completed"
            execution.completed_at = datetime.now()
            
            # Calculate actual P&L
            self._calculate_actual_pnl(execution)
            
            # Update performance tracking
            self.performance_tracker.record_execution(execution)
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            execution.status = "failed"
            
            # Cancel any open orders
            await self._cancel_execution_orders(execution)
        
        return execution
    
    def _determine_execution_strategy(self, opportunity: ArbitrageOpportunity) -> ExecutionStrategy:
        """Determine optimal execution strategy"""
        
        # For small orders, be aggressive
        if opportunity.max_quantity < Decimal("1000"):
            return ExecutionStrategy.AGGRESSIVE
        
        # For large orders with low risk, use stealth
        if opportunity.max_quantity > Decimal("10000") and opportunity.execution_risk < 0.3:
            return ExecutionStrategy.STEALTH
        
        # For medium risk, use mixed approach
        if opportunity.execution_risk > 0.3 and opportunity.execution_risk < 0.6:
            return ExecutionStrategy.MIXED
        
        # Default to passive for everything else
        return ExecutionStrategy.PASSIVE
    
    async def _execute_aggressive(self, execution: ArbitrageExecution):
        """Execute with aggressive strategy (market orders)"""
        opportunity = execution.opportunity
        
        # Place market orders simultaneously
        buy_task = self.execution_engine.place_order(
            exchange=opportunity.buy_venue,
            symbol=opportunity.symbol,
            side='buy',
            order_type='market',
            quantity=opportunity.max_quantity
        )
        
        sell_task = self.execution_engine.place_order(
            exchange=opportunity.sell_venue,
            symbol=opportunity.symbol,
            side='sell',
            order_type='market',
            quantity=opportunity.max_quantity
        )
        
        # Execute simultaneously
        buy_order, sell_order = await asyncio.gather(buy_task, sell_task)
        
        execution.buy_orders.append(buy_order)
        execution.sell_orders.append(sell_order)
        
        # Wait for fills
        await self._wait_for_fills(execution)
```

### 2. Advanced Execution Engine

```python
# src/arbitrage/execution_engine.py
class ExecutionEngine:
    """Handles order execution across multiple venues"""
    
    def __init__(self, exchanges: Dict[str, Any]):
        self.exchanges = exchanges
        self.order_manager = OrderManager()
        self.latency_monitor = LatencyMonitor()
        self.logger = ComponentLogger("ExecutionEngine", "arbitrage")
    
    async def place_order(self, exchange: str, symbol: str, side: str,
                         order_type: str, quantity: Decimal,
                         price: Optional[Decimal] = None) -> Dict:
        """Place order with latency tracking"""
        
        start_time = time.time_ns()
        
        try:
            # Get exchange instance
            exchange_client = self.exchanges[exchange]
            
            # Convert order type
            if order_type == 'market':
                order = await exchange_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    amount=quantity
                )
            else:
                order = await exchange_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    amount=quantity,
                    price=price
                )
            
            # Track latency
            latency_ns = time.time_ns() - start_time
            self.latency_monitor.record_latency(exchange, 'order_placement', latency_ns)
            
            # Track order
            self.order_manager.add_order(order)
            
            return {
                'order': order,
                'latency_ms': latency_ns / 1_000_000,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Order placement failed on {exchange}: {e}")
            raise
    
    async def place_ioc_order(self, exchange: str, symbol: str, side: str,
                             quantity: Decimal, price: Decimal) -> Dict:
        """Place Immediate-Or-Cancel order"""
        # IOC orders for better control
        pass
    
    async def place_iceberg_order(self, exchange: str, symbol: str, side: str,
                                 total_quantity: Decimal, visible_quantity: Decimal,
                                 price: Decimal) -> List[Dict]:
        """Place iceberg order to minimize market impact"""
        orders = []
        remaining = total_quantity
        
        while remaining > 0:
            chunk = min(remaining, visible_quantity)
            
            order = await self.place_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type='limit',
                quantity=chunk,
                price=price
            )
            
            orders.append(order)
            remaining -= chunk
            
            # Small delay between chunks
            await asyncio.sleep(0.1)
        
        return orders
    
    async def cancel_order(self, exchange: str, order_id: str, symbol: str) -> bool:
        """Cancel order with latency tracking"""
        start_time = time.time_ns()
        
        try:
            result = await self.exchanges[exchange].cancel_order(order_id, symbol)
            
            latency_ns = time.time_ns() - start_time
            self.latency_monitor.record_latency(exchange, 'order_cancel', latency_ns)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order cancel failed: {e}")
            return False

class SmartOrderRouter:
    """Intelligent order routing across venues"""
    
    def __init__(self, exchanges: Dict[str, Any]):
        self.exchanges = exchanges
        self.venue_analyzer = VenueAnalyzer()
        self.cost_model = TradingCostModel()
        self.logger = ComponentLogger("SmartOrderRouter", "arbitrage")
    
    async def route_order(self, symbol: str, side: str, quantity: Decimal,
                         urgency: str = "normal") -> List[Dict]:
        """Route order optimally across venues"""
        
        # Analyze venues
        venue_analysis = await self.venue_analyzer.analyze_venues(
            symbol, side, quantity, self.exchanges
        )
        
        # Calculate optimal routing
        routing_plan = self._calculate_routing(
            venue_analysis, quantity, urgency
        )
        
        # Execute routing plan
        executions = []
        for venue, allocation in routing_plan.items():
            if allocation['quantity'] > 0:
                execution = await self._execute_venue_order(
                    venue, symbol, side, allocation
                )
                executions.append(execution)
        
        return executions
    
    def _calculate_routing(self, venue_analysis: Dict, 
                          total_quantity: Decimal, urgency: str) -> Dict:
        """Calculate optimal order routing"""
        
        routing_plan = {}
        remaining = total_quantity
        
        # Sort venues by execution quality
        sorted_venues = sorted(
            venue_analysis.items(),
            key=lambda x: x[1]['execution_score'],
            reverse=True
        )
        
        for venue, analysis in sorted_venues:
            if remaining <= 0:
                break
            
            # Calculate allocation
            if urgency == "high":
                # Take more from best venues
                allocation_pct = 0.7 if analysis['execution_score'] > 0.8 else 0.3
            else:
                # Distribute more evenly
                allocation_pct = 1.0 / len(sorted_venues)
            
            allocated_qty = min(
                remaining,
                total_quantity * Decimal(str(allocation_pct)),
                Decimal(str(analysis['available_liquidity']))
            )
            
            if allocated_qty > 0:
                routing_plan[venue] = {
                    'quantity': allocated_qty,
                    'strategy': 'aggressive' if urgency == "high" else 'passive',
                    'expected_price': analysis['expected_price']
                }
                
                remaining -= allocated_qty
        
        return routing_plan

class VenueAnalyzer:
    """Analyzes trading venues for optimal execution"""
    
    async def analyze_venues(self, symbol: str, side: str, 
                           quantity: Decimal, exchanges: Dict) -> Dict:
        """Analyze all venues for execution quality"""
        
        analyses = {}
        
        for venue, exchange in exchanges.items():
            try:
                # Get order book
                order_book = await exchange.get_order_book(symbol, limit=20)
                
                # Analyze liquidity
                liquidity_analysis = self._analyze_liquidity(
                    order_book, side, quantity
                )
                
                # Get recent trades
                trades = await exchange.get_trades(symbol, limit=100)
                
                # Analyze execution quality
                execution_analysis = self._analyze_execution_quality(
                    trades, order_book
                )
                
                analyses[venue] = {
                    **liquidity_analysis,
                    **execution_analysis,
                    'execution_score': self._calculate_execution_score(
                        liquidity_analysis, execution_analysis
                    )
                }
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {venue}: {e}")
                analyses[venue] = {
                    'available_liquidity': 0,
                    'expected_price': 0,
                    'execution_score': 0
                }
        
        return analyses
    
    def _analyze_liquidity(self, order_book: Dict, side: str, 
                          quantity: Decimal) -> Dict:
        """Analyze order book liquidity"""
        
        book_side = 'asks' if side == 'buy' else 'bids'
        levels = order_book[book_side]
        
        cumulative_qty = Decimal("0")
        weighted_price = Decimal("0")
        
        for price, qty in levels:
            price = Decimal(str(price))
            qty = Decimal(str(qty))
            
            if cumulative_qty + qty >= quantity:
                # Partial fill at this level
                remaining = quantity - cumulative_qty
                weighted_price += price * remaining
                cumulative_qty = quantity
                break
            else:
                # Full level consumed
                weighted_price += price * qty
                cumulative_qty += qty
        
        if cumulative_qty > 0:
            avg_price = weighted_price / cumulative_qty
        else:
            avg_price = Decimal("0")
        
        return {
            'available_liquidity': float(cumulative_qty),
            'expected_price': float(avg_price),
            'spread': float(Decimal(str(levels[0][0])) - Decimal(str(order_book['bids'][0][0]))) if order_book['bids'] else 0,
            'depth_levels': len(levels)
        }
```

### 3. Risk Management System

```python
# src/arbitrage/risk_management.py
class ArbitrageRiskManager:
    """Risk management for arbitrage operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits
        self.max_position_value = Decimal(str(config.get('max_position_value', 1000000)))
        self.max_single_trade = Decimal(str(config.get('max_single_trade', 100000)))
        self.max_daily_loss = Decimal(str(config.get('max_daily_loss', 10000)))
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.7)
        
        # Tracking
        self.positions = defaultdict(Decimal)
        self.daily_pnl = Decimal("0")
        self.risk_metrics = {}
        
        self.logger = ComponentLogger("ArbitrageRiskManager", "risk")
    
    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate if opportunity meets risk criteria"""
        
        # Check single trade limit
        trade_value = opportunity.max_quantity * opportunity.buy_price
        if trade_value > self.max_single_trade:
            self.logger.warning(f"Trade size {trade_value} exceeds limit")
            return False
        
        # Check position limits
        current_position = self.positions[opportunity.symbol]
        new_position = current_position + opportunity.max_quantity
        
        if abs(new_position * opportunity.buy_price) > self.max_position_value:
            self.logger.warning(f"Position limit exceeded for {opportunity.symbol}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            self.logger.warning("Daily loss limit reached")
            return False
        
        # Check execution risk
        if opportunity.execution_risk > 0.7:
            self.logger.warning(f"High execution risk: {opportunity.execution_risk}")
            return False
        
        # Check confidence
        if opportunity.confidence_score < 0.6:
            self.logger.warning(f"Low confidence: {opportunity.confidence_score}")
            return False
        
        return True
    
    def check_limits(self, opportunity: ArbitrageOpportunity) -> bool:
        """Real-time limit checking"""
        # Additional real-time checks
        return self.validate_opportunity(opportunity)
    
    def update_position(self, symbol: str, quantity: Decimal, side: str):
        """Update position tracking"""
        if side == 'buy':
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] -= quantity
    
    def update_pnl(self, pnl: Decimal):
        """Update daily P&L"""
        self.daily_pnl += pnl
    
    def calculate_var(self, confidence_level: float = 0.95) -> Decimal:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        position_values = []
        
        for symbol, quantity in self.positions.items():
            # Get current price (would fetch from market data)
            price = Decimal("1000")  # Placeholder
            position_values.append(float(quantity * price))
        
        if not position_values:
            return Decimal("0")
        
        # Historical VaR calculation
        volatility = 0.02  # 2% daily volatility assumption
        portfolio_value = sum(position_values)
        
        # Normal distribution VaR
        z_score = 1.645 if confidence_level == 0.95 else 2.326
        var = Decimal(str(portfolio_value * volatility * z_score))
        
        return var
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        
        total_position_value = Decimal("0")
        for symbol, quantity in self.positions.items():
            # Would fetch real prices
            price = Decimal("1000")
            total_position_value += abs(quantity * price)
        
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        
        return {
            'total_position_value': float(total_position_value),
            'position_utilization': float(total_position_value / self.max_position_value),
            'daily_pnl': float(self.daily_pnl),
            'daily_loss_utilization': float(-self.daily_pnl / self.max_daily_loss) if self.daily_pnl < 0 else 0,
            'var_95': float(var_95),
            'var_99': float(var_99),
            'position_count': len([p for p in self.positions.values() if p != 0])
        }

class PositionTracker:
    """Tracks positions across all venues"""
    
    def __init__(self):
        self.positions = defaultdict(lambda: defaultdict(Decimal))
        self.pending_orders = defaultdict(list)
        self.filled_orders = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def update_position(self, venue: str, symbol: str, 
                            quantity: Decimal, side: str):
        """Update position for venue and symbol"""
        async with self.lock:
            if side == 'buy':
                self.positions[venue][symbol] += quantity
            else:
                self.positions[venue][symbol] -= quantity
    
    def get_position(self, symbol: str) -> Decimal:
        """Get total position across all venues"""
        total = Decimal("0")
        for venue_positions in self.positions.values():
            total += venue_positions.get(symbol, Decimal("0"))
        return total
    
    def get_venue_position(self, venue: str, symbol: str) -> Decimal:
        """Get position for specific venue"""
        return self.positions[venue].get(symbol, Decimal("0"))
    
    def get_all_positions(self) -> Dict[str, Dict[str, Decimal]]:
        """Get all positions by venue"""
        return dict(self.positions)
```

### 4. Latency Optimization

```python
# src/arbitrage/latency_optimization.py
class LatencyMonitor:
    """Monitors and optimizes latency across venues"""
    
    def __init__(self):
        self.latency_history = defaultdict(lambda: defaultdict(list))
        self.latency_stats = defaultdict(dict)
        self.optimization_engine = LatencyOptimizer()
        
    def record_latency(self, venue: str, operation: str, latency_ns: int):
        """Record latency measurement"""
        latency_ms = latency_ns / 1_000_000
        
        # Add to history
        self.latency_history[venue][operation].append({
            'timestamp': time.time_ns(),
            'latency_ms': latency_ms
        })
        
        # Keep only recent data (last 1000 measurements)
        if len(self.latency_history[venue][operation]) > 1000:
            self.latency_history[venue][operation].pop(0)
        
        # Update statistics
        self._update_stats(venue, operation)
    
    def _update_stats(self, venue: str, operation: str):
        """Update latency statistics"""
        measurements = [m['latency_ms'] for m in self.latency_history[venue][operation]]
        
        if measurements:
            self.latency_stats[venue][operation] = {
                'mean': np.mean(measurements),
                'median': np.median(measurements),
                'p95': np.percentile(measurements, 95),
                'p99': np.percentile(measurements, 99),
                'std': np.std(measurements),
                'min': np.min(measurements),
                'max': np.max(measurements)
            }
    
    def get_venue_latency(self, venue: str) -> Dict[str, Any]:
        """Get latency statistics for a venue"""
        return self.latency_stats.get(venue, {})
    
    def get_fastest_venue(self, operation: str) -> Optional[str]:
        """Get venue with lowest latency for operation"""
        fastest_venue = None
        lowest_latency = float('inf')
        
        for venue, operations in self.latency_stats.items():
            if operation in operations:
                mean_latency = operations[operation]['mean']
                if mean_latency < lowest_latency:
                    lowest_latency = mean_latency
                    fastest_venue = venue
        
        return fastest_venue

class LatencyOptimizer:
    """Optimizes execution based on latency patterns"""
    
    def __init__(self):
        self.connection_pool = ConnectionPool()
        self.route_optimizer = RouteOptimizer()
        
    async def optimize_connection(self, venue: str, latency_stats: Dict):
        """Optimize connection to venue based on latency"""
        
        # Check if we should adjust connection parameters
        if latency_stats.get('order_placement', {}).get('p95', 0) > 100:
            # High latency - optimize connection
            await self.connection_pool.optimize_venue_connection(venue)
    
    def calculate_optimal_timing(self, venues: List[str], 
                               latency_data: Dict) -> Dict[str, float]:
        """Calculate optimal order timing for simultaneous execution"""
        
        # Find the slowest venue
        max_latency = 0
        for venue in venues:
            venue_latency = latency_data.get(venue, {}).get('order_placement', {}).get('mean', 0)
            max_latency = max(max_latency, venue_latency)
        
        # Calculate delays for faster venues
        timing = {}
        for venue in venues:
            venue_latency = latency_data.get(venue, {}).get('order_placement', {}).get('mean', 0)
            delay = max(0, max_latency - venue_latency)
            timing[venue] = delay / 1000  # Convert to seconds
        
        return timing

class ConnectionPool:
    """Manages optimized connections to venues"""
    
    def __init__(self):
        self.connections = {}
        self.connection_metrics = defaultdict(dict)
        
    async def optimize_venue_connection(self, venue: str):
        """Optimize connection parameters for a venue"""
        
        # Adjust connection parameters
        if venue in self.connections:
            conn = self.connections[venue]
            
            # Increase connection pool size
            if hasattr(conn, 'pool_size'):
                conn.pool_size = min(conn.pool_size * 2, 100)
            
            # Enable TCP optimizations
            if hasattr(conn, 'socket_options'):
                conn.socket_options.update({
                    'TCP_NODELAY': 1,  # Disable Nagle's algorithm
                    'SO_KEEPALIVE': 1,  # Enable keep-alive
                    'TCP_QUICKACK': 1   # Quick ACK mode
                })
```

### 5. Performance Analytics

```python
# src/arbitrage/performance_analytics.py
class PerformanceTracker:
    """Tracks arbitrage performance metrics"""
    
    def __init__(self):
        self.executions = []
        self.daily_metrics = defaultdict(dict)
        self.opportunity_metrics = defaultdict(lambda: {
            'discovered': 0,
            'executed': 0,
            'successful': 0,
            'failed': 0,
            'total_profit': Decimal("0"),
            'total_loss': Decimal("0")
        })
        
    def record_execution(self, execution: ArbitrageExecution):
        """Record execution results"""
        self.executions.append(execution)
        
        # Update opportunity metrics
        venue_pair = f"{execution.opportunity.buy_venue}-{execution.opportunity.sell_venue}"
        metrics = self.opportunity_metrics[venue_pair]
        
        metrics['executed'] += 1
        
        if execution.status == 'completed':
            metrics['successful'] += 1
            if execution.actual_profit > 0:
                metrics['total_profit'] += execution.actual_profit
            else:
                metrics['total_loss'] += abs(execution.actual_profit)
        else:
            metrics['failed'] += 1
        
        # Update daily metrics
        self._update_daily_metrics(execution)
    
    def _update_daily_metrics(self, execution: ArbitrageExecution):
        """Update daily performance metrics"""
        date = execution.started_at.date()
        
        if date not in self.daily_metrics:
            self.daily_metrics[date] = {
                'executions': 0,
                'successful': 0,
                'failed': 0,
                'gross_profit': Decimal("0"),
                'fees': Decimal("0"),
                'net_profit': Decimal("0"),
                'volume': Decimal("0")
            }
        
        metrics = self.daily_metrics[date]
        metrics['executions'] += 1
        
        if execution.status == 'completed':
            metrics['successful'] += 1
            metrics['gross_profit'] += execution.actual_sell_revenue - execution.actual_buy_cost
            metrics['fees'] += execution.actual_fees
            metrics['net_profit'] += execution.actual_profit
            metrics['volume'] += execution.executed_quantity * execution.opportunity.buy_price
        else:
            metrics['failed'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        
        total_executions = len(self.executions)
        successful = sum(1 for e in self.executions if e.status == 'completed')
        
        total_profit = sum(e.actual_profit for e in self.executions 
                          if e.status == 'completed' and e.actual_profit > 0)
        total_loss = sum(abs(e.actual_profit) for e in self.executions 
                        if e.status == 'completed' and e.actual_profit < 0)
        
        avg_execution_time = np.mean([
            (e.completed_at - e.started_at).total_seconds() 
            for e in self.executions 
            if e.completed_at
        ]) if self.executions else 0
        
        return {
            'total_executions': total_executions,
            'success_rate': successful / total_executions if total_executions > 0 else 0,
            'total_profit': float(total_profit),
            'total_loss': float(total_loss),
            'net_pnl': float(total_profit - total_loss),
            'avg_execution_time_ms': avg_execution_time * 1000,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of arbitrage strategy"""
        if not self.daily_metrics:
            return 0.0
        
        daily_returns = []
        for date, metrics in sorted(self.daily_metrics.items()):
            daily_returns.append(float(metrics['net_profit']))
        
        if len(daily_returns) < 2:
            return 0.0
        
        returns_array = np.array(daily_returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming 365 trading days)
        return (avg_return / std_return) * np.sqrt(365)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.executions:
            return 0.0
        
        cumulative_pnl = []
        running_total = Decimal("0")
        
        for execution in sorted(self.executions, key=lambda x: x.started_at):
            if execution.status == 'completed':
                running_total += execution.actual_profit
                cumulative_pnl.append(float(running_total))
        
        if not cumulative_pnl:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for value in cumulative_pnl[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        
        return max_dd
```

### 6. Testing Framework

```python
# tests/unit/test_step13_arbitrage.py
class TestArbitrageEngine:
    """Test arbitrage engine functionality"""
    
    async def test_opportunity_detection(self):
        """Test arbitrage opportunity detection"""
        
        # Create test engine
        exchanges = {
            'exchange1': MockExchange('exchange1'),
            'exchange2': MockExchange('exchange2')
        }
        
        config = {
            'symbols': ['BTC/USDT'],
            'min_profit_usd': 10,
            'exchanges': {
                'exchange1': {'taker_fee': 0.001},
                'exchange2': {'taker_fee': 0.001}
            },
            'risk': {}
        }
        
        engine = CrossExchangeArbitrageEngine(exchanges, config)
        
        # Set different prices
        engine.order_books['BTC/USDT']['exchange1'] = {
            'bids': [[40000, 1]],
            'asks': [[40100, 1]],
            'timestamp': time.time() * 1000
        }
        
        engine.order_books['BTC/USDT']['exchange2'] = {
            'bids': [[40200, 1]],
            'asks': [[40300, 1]],
            'timestamp': time.time() * 1000
        }
        
        # Detect opportunities
        await engine._discover_opportunities('BTC/USDT')
        
        # Should find opportunity
        assert len(engine.opportunities) > 0
        
        # Check opportunity details
        _, opportunity = engine.opportunities.peekitem(-1)
        assert opportunity.buy_venue == 'exchange1'
        assert opportunity.sell_venue == 'exchange2'
        assert opportunity.net_profit > 0
    
    def test_fee_calculation(self):
        """Test fee calculation accuracy"""
        engine = CrossExchangeArbitrageEngine({}, {
            'exchanges': {
                'binance': {'taker_fee': 0.001},
                'coinbase': {'taker_fee': 0.0025}
            }
        })
        
        # Mock order books
        engine.order_books['ETH/USDT'] = {
            'binance': {
                'asks': [[2000, 10]],
                'bids': [[1999, 10]]
            },
            'coinbase': {
                'asks': [[2010, 10]],
                'bids': [[2009, 10]]
            }
        }
        
        fees = engine._calculate_fees('binance', 'coinbase', 'ETH/USDT', Decimal("1"))
        
        # Buy fee: 1 * 2000 * 0.001 = 2
        # Sell fee: 1 * 2009 * 0.0025 = 5.0225
        expected_fees = Decimal("2") + Decimal("5.0225")
        
        assert abs(fees - expected_fees) < Decimal("0.01")
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        engine = CrossExchangeArbitrageEngine({}, {})
        
        # Fresh, deep order book
        good_ob = {
            'bids': [[100, 10], [99.9, 20], [99.8, 30]],
            'asks': [[100.1, 10], [100.2, 20], [100.3, 30]],
            'timestamp': time.time() * 1000
        }
        
        confidence = engine._calculate_confidence(good_ob, good_ob)
        assert confidence > 0.8
        
        # Old, shallow order book
        bad_ob = {
            'bids': [[100, 1]],
            'asks': [[101, 1]],
            'timestamp': (time.time() - 10) * 1000  # 10 seconds old
        }
        
        confidence = engine._calculate_confidence(bad_ob, bad_ob)
        assert confidence < 0.6

class TestExecutionEngine:
    """Test execution engine"""
    
    async def test_smart_order_routing(self):
        """Test smart order routing"""
        exchanges = {
            'venue1': MockExchange('venue1'),
            'venue2': MockExchange('venue2'),
            'venue3': MockExchange('venue3')
        }
        
        router = SmartOrderRouter(exchanges)
        
        # Route a large order
        routing = await router.route_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal("10"),
            urgency='normal'
        )
        
        # Should split across venues
        assert len(routing) >= 2
        
        # Total should match
        total_routed = sum(r['order']['amount'] for r in routing)
        assert total_routed == Decimal("10")
    
    async def test_latency_monitoring(self):
        """Test latency monitoring"""
        monitor = LatencyMonitor()
        
        # Record some latencies
        for i in range(100):
            monitor.record_latency('binance', 'order_placement', 
                                 (20 + np.random.normal(0, 5)) * 1_000_000)  # 20ms ± 5ms
            monitor.record_latency('coinbase', 'order_placement',
                                 (30 + np.random.normal(0, 8)) * 1_000_000)  # 30ms ± 8ms
        
        # Check statistics
        binance_stats = monitor.get_venue_latency('binance')
        assert 15 < binance_stats['order_placement']['mean'] < 25
        
        # Check fastest venue
        fastest = monitor.get_fastest_venue('order_placement')
        assert fastest == 'binance'

class TestRiskManagement:
    """Test risk management system"""
    
    def test_position_limits(self):
        """Test position limit enforcement"""
        config = {
            'max_position_value': 100000,
            'max_single_trade': 10000,
            'max_daily_loss': 1000
        }
        
        risk_manager = ArbitrageRiskManager(config)
        
        # Create opportunity within limits
        good_opp = ArbitrageOpportunity(
            opportunity_id="test1",
            type=ArbitrageType.CROSS_EXCHANGE,
            symbol="BTC/USDT",
            buy_venue="binance",
            sell_venue="coinbase",
            buy_price=Decimal("40000"),
            sell_price=Decimal("40100"),
            max_quantity=Decimal("0.2"),
            gross_profit=Decimal("20"),
            net_profit=Decimal("15"),
            profit_percentage=Decimal("0.0375"),
            confidence_score=0.8,
            execution_risk=0.2,
            discovered_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=5)
        )
        
        assert risk_manager.validate_opportunity(good_opp) == True
        
        # Create opportunity exceeding limits
        bad_opp = ArbitrageOpportunity(
            opportunity_id="test2",
            type=ArbitrageType.CROSS_EXCHANGE,
            symbol="BTC/USDT",
            buy_venue="binance",
            sell_venue="coinbase",
            buy_price=Decimal("40000"),
            sell_price=Decimal("40100"),
            max_quantity=Decimal("5"),  # Too large
            gross_profit=Decimal("500"),
            net_profit=Decimal("400"),
            profit_percentage=Decimal("0.025"),
            confidence_score=0.8,
            execution_risk=0.2,
            discovered_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=5)
        )
        
        assert risk_manager.validate_opportunity(bad_opp) == False
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        risk_manager = ArbitrageRiskManager({})
        
        # Add some positions
        risk_manager.positions['BTC/USDT'] = Decimal("1")
        risk_manager.positions['ETH/USDT'] = Decimal("10")
        
        var_95 = risk_manager.calculate_var(0.95)
        var_99 = risk_manager.calculate_var(0.99)
        
        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher
```

### 7. Integration Tests

```python
# tests/integration/test_step13_arbitrage_integration.py
async def test_complete_arbitrage_workflow():
    """Test complete arbitrage discovery and execution"""
    
    # Setup
    exchanges = {
        'binance': create_mock_exchange('binance'),
        'coinbase': create_mock_exchange('coinbase')
    }
    
    config = {
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'min_profit_usd': 10,
        'exchanges': {
            'binance': {'taker_fee': 0.001},
            'coinbase': {'taker_fee': 0.0025}
        },
        'risk': {
            'max_position_value': 100000,
            'max_single_trade': 10000
        }
    }
    
    engine = CrossExchangeArbitrageEngine(exchanges, config)
    
    # Create price discrepancy
    exchanges['binance'].set_order_book('BTC/USDT', {
        'bids': [[39900, 1]],
        'asks': [[40000, 1]]
    })
    
    exchanges['coinbase'].set_order_book('BTC/USDT', {
        'bids': [[40100, 1]],
        'asks': [[40200, 1]]
    })
    
    # Start engine
    engine_task = asyncio.create_task(engine.start())
    
    # Wait for opportunity discovery
    await asyncio.sleep(0.5)
    
    # Should have discovered opportunity
    assert len(engine.opportunities) > 0
    
    # Wait for execution
    await asyncio.sleep(1)
    
    # Check execution completed
    assert len(engine.active_executions) > 0
    
    execution = list(engine.active_executions.values())[0]
    assert execution.status == 'completed'
    assert execution.actual_profit > 0
    
    # Cleanup
    engine_task.cancel()

async def test_multi_symbol_arbitrage():
    """Test arbitrage across multiple symbols"""
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'MATIC/USDT']
    exchanges = {
        'binance': create_mock_exchange('binance'),
        'coinbase': create_mock_exchange('coinbase'),
        'kraken': create_mock_exchange('kraken')
    }
    
    # Create different price discrepancies
    test_data = [
        ('BTC/USDT', 'binance', 40000, 'coinbase', 40100),
        ('ETH/USDT', 'coinbase', 2000, 'kraken', 2020),
        ('SOL/USDT', 'kraken', 100, 'binance', 102)
    ]
    
    for symbol, buy_ex, buy_price, sell_ex, sell_price in test_data:
        exchanges[buy_ex].set_order_book(symbol, {
            'bids': [[buy_price - 10, 10]],
            'asks': [[buy_price, 10]]
        })
        exchanges[sell_ex].set_order_book(symbol, {
            'bids': [[sell_price, 10]],
            'asks': [[sell_price + 10, 10]]
        })
    
    # Run arbitrage detection
    engine = CrossExchangeArbitrageEngine(exchanges, {
        'symbols': symbols,
        'min_profit_usd': 5,
        'exchanges': {ex: {'taker_fee': 0.001} for ex in exchanges},
        'risk': {}
    })
    
    # Discover opportunities
    for symbol in symbols:
        await engine._discover_opportunities(symbol)
    
    # Should find multiple opportunities
    assert len(engine.opportunities) >= 2
```

### 8. System Tests

```python
# tests/system/test_step13_production_arbitrage.py
async def test_high_frequency_arbitrage():
    """Test high-frequency arbitrage performance"""
    
    # Production-like setup
    engine = create_production_arbitrage_engine()
    
    # Performance counters
    opportunities_found = 0
    executions_completed = 0
    total_profit = Decimal("0")
    start_time = time.time()
    
    # Run for 60 seconds
    async def monitor_performance():
        nonlocal opportunities_found, executions_completed, total_profit
        
        while time.time() - start_time < 60:
            opportunities_found = len(engine.opportunities)
            
            for execution in engine.active_executions.values():
                if execution.status == 'completed':
                    executions_completed += 1
                    total_profit += execution.actual_profit
            
            await asyncio.sleep(0.1)
    
    # Start engine and monitoring
    engine_task = asyncio.create_task(engine.start())
    monitor_task = asyncio.create_task(monitor_performance())
    
    # Wait for test duration
    await asyncio.sleep(60)
    
    # Check performance metrics
    duration = time.time() - start_time
    
    assert opportunities_found > 100  # Should find many opportunities
    assert executions_completed > 10  # Should execute some
    assert total_profit > 0  # Should be profitable
    
    # Check latency
    avg_execution_time = engine.performance_tracker.get_performance_summary()['avg_execution_time_ms']
    assert avg_execution_time < 100  # Sub-100ms execution
    
    # Cleanup
    engine_task.cancel()
    monitor_task.cancel()

async def test_failover_resilience():
    """Test system resilience to exchange failures"""
    
    exchanges = {
        'primary': create_mock_exchange('primary'),
        'backup1': create_mock_exchange('backup1'),
        'backup2': create_mock_exchange('backup2')
    }
    
    engine = CrossExchangeArbitrageEngine(exchanges, load_config())
    
    # Start engine
    engine_task = asyncio.create_task(engine.start())
    
    # Simulate primary exchange failure
    await asyncio.sleep(5)
    exchanges['primary'].simulate_failure()
    
    # Engine should continue with other exchanges
    await asyncio.sleep(5)
    
    # Check that arbitrage still works
    opportunities_after_failure = len(engine.opportunities)
    assert opportunities_after_failure > 0
    
    # Restore primary exchange
    exchanges['primary'].restore()
    
    await asyncio.sleep(5)
    
    # Should resume using all exchanges
    active_venues = set()
    for opp in engine.opportunities.values():
        active_venues.add(opp.buy_venue)
        active_venues.add(opp.sell_venue)
    
    assert 'primary' in active_venues
    
    engine_task.cancel()

async def test_risk_limit_enforcement():
    """Test risk limits under stress"""
    
    engine = create_production_arbitrage_engine()
    
    # Set aggressive risk limits
    engine.risk_manager.max_position_value = Decimal("50000")
    engine.risk_manager.max_daily_loss = Decimal("1000")
    
    # Create many high-value opportunities
    for i in range(100):
        opportunity = ArbitrageOpportunity(
            opportunity_id=f"stress_test_{i}",
            type=ArbitrageType.CROSS_EXCHANGE,
            symbol="BTC/USDT",
            buy_venue="venue1",
            sell_venue="venue2",
            buy_price=Decimal("40000"),
            sell_price=Decimal("40100"),
            max_quantity=Decimal("2"),  # Large size
            gross_profit=Decimal("200"),
            net_profit=Decimal("150"),
            profit_percentage=Decimal("0.375"),
            confidence_score=0.9,
            execution_risk=0.1,
            discovered_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=60)
        )
        
        engine._add_opportunity(opportunity)
    
    # Run execution loop
    execution_count = 0
    for _ in range(100):
        if engine.opportunities:
            if engine._should_execute(engine.opportunities.peekitem(-1)[1]):
                execution_count += 1
                engine.opportunities.popitem(-1)
    
    # Should have stopped due to position limits
    assert execution_count < 100
    
    # Check position value
    total_position = engine.risk_manager.positions['BTC/USDT']
    assert total_position * Decimal("40000") <= engine.risk_manager.max_position_value
```

## ✅ Validation Checklist

### Core Engine
- [ ] Opportunity discovery working across venues
- [ ] Profit calculation accurate with fees
- [ ] Risk validation functioning
- [ ] Execution engine operational
- [ ] Position tracking accurate

### Performance
- [ ] Sub-100ms opportunity detection
- [ ] Concurrent order execution
- [ ] Latency monitoring active
- [ ] Smart order routing working
- [ ] Market impact minimized

### Risk Management
- [ ] Position limits enforced
- [ ] Daily loss limits working
- [ ] VaR calculation accurate
- [ ] Correlation exposure tracked
- [ ] Emergency stops functional

### Optimization
- [ ] Venue selection optimized
- [ ] Order timing synchronized
- [ ] Connection pooling efficient
- [ ] Memory usage controlled
- [ ] CPU usage optimized

### Analytics
- [ ] P&L tracking accurate
- [ ] Performance metrics calculated
- [ ] Sharpe ratio meaningful
- [ ] Execution analysis working
- [ ] Venue comparison available

## 📊 Performance Benchmarks

### Latency Targets
- Opportunity detection: < 10ms
- Order placement: < 50ms
- Total execution: < 100ms
- Market data processing: < 5ms

### Throughput
- Opportunities/second: 1000+
- Executions/minute: 100+
- Concurrent positions: 50+
- Data points/second: 10,000+

### Success Metrics
- Execution success rate: > 90%
- Positive P&L ratio: > 70%
- Average slippage: < 0.1%
- Sharpe ratio: > 2.0

## 🐛 Common Issues

1. **Latency Spikes**
   - Monitor network conditions
   - Use connection pooling
   - Implement circuit breakers
   - Have backup routes

2. **Execution Failures**
   - Implement retry logic
   - Use IOC orders
   - Monitor fill rates
   - Adjust timing dynamically

3. **Risk Breaches**
   - Real-time position monitoring
   - Pre-trade validation
   - Automatic hedging
   - Emergency liquidation

## 🎯 Success Criteria

Step 13 is complete when:
1. ✅ Multi-venue arbitrage engine operational
2. ✅ Low-latency execution achieved
3. ✅ Risk management comprehensive
4. ✅ Performance tracking accurate
5. ✅ System resilient to failures

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 14: Machine Learning Models](step-14-ml-models.md)

## 📚 Additional Resources

- [High-Frequency Trading Guide](../references/hft-guide.md)
- [Market Microstructure Theory](../references/microstructure-theory.md)
- [Latency Optimization Techniques](../references/latency-techniques.md)
- [Arbitrage Strategy Backtesting](../references/arbitrage-backtesting.md)