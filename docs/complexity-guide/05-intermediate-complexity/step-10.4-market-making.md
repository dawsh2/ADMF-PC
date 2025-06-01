# Step 10.4: Market Making

**Status**: Intermediate Complexity Step
**Complexity**: Very High
**Prerequisites**: [Step 10.3: Execution Algorithms](step-10.3-execution-algos.md) completed
**Architecture Ref**: [Market Making Architecture](../architecture/market-making-architecture.md)

## 🎯 Objective

Implement automated market making strategies:
- Optimal bid-ask spread pricing
- Inventory risk management and hedging
- Dynamic quote adjustment algorithms
- Adverse selection detection and mitigation
- Multi-level order book strategies
- Cross-venue arbitrage opportunities

## 📋 Required Reading

Before starting:
1. [Market Making Theory](../references/market-making-theory.md)
2. [Inventory Risk Management](../references/inventory-risk.md)
3. [Optimal Market Making](../references/optimal-mm.md)

## 🏗️ Implementation Tasks

### 1. Core Market Making Framework

```python
# src/market_making/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class QuoteState(Enum):
    """Market maker quote states"""
    ACTIVE = "active"
    PASSIVE = "passive" 
    AGGRESSIVE = "aggressive"
    WITHDRAWN = "withdrawn"
    INVENTORY_LIMIT = "inventory_limit"

@dataclass
class MarketMakerPosition:
    """Current position state for market maker"""
    asset: str
    quantity: float
    vwap: float
    unrealized_pnl: float
    
    # Inventory limits
    max_long: float
    max_short: float
    
    # Risk metrics
    inventory_risk: float
    time_in_position: timedelta
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1
    
    @property
    def inventory_ratio(self) -> float:
        """Position as ratio of max capacity"""
        if self.quantity > 0:
            return self.quantity / self.max_long
        elif self.quantity < 0:
            return abs(self.quantity) / self.max_short
        return 0.0

@dataclass
class QuoteParameters:
    """Parameters for quote generation"""
    fair_value: float
    volatility: float
    spread: float
    
    # Skew parameters
    inventory_skew: float = 0.0
    flow_skew: float = 0.0
    vol_skew: float = 0.0
    
    # Size parameters
    base_size: int = 100
    size_multiplier: float = 1.0
    
    # Risk parameters
    max_spread: float = 0.01
    min_spread: float = 0.0001

@dataclass
class Quote:
    """Bid/ask quote pair"""
    asset: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    
    timestamp: datetime
    quote_id: str
    
    # Metadata
    fair_value: float
    spread: float
    skew: float
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread_bps(self) -> float:
        return (self.ask_price - self.bid_price) / self.mid_price * 10000

class MarketMaker(ABC):
    """Base class for market making strategies"""
    
    def __init__(self, asset: str, config: Dict):
        self.asset = asset
        self.config = config
        
        # Position tracking
        self.position = MarketMakerPosition(
            asset=asset,
            quantity=0,
            vwap=0,
            unrealized_pnl=0,
            max_long=config.get('max_long', 10000),
            max_short=config.get('max_short', 10000),
            inventory_risk=0,
            time_in_position=timedelta()
        )
        
        # Quote tracking
        self.current_quotes: Dict[str, Quote] = {}
        self.quote_history: List[Quote] = []
        
        # Risk management
        self.risk_manager = InventoryRiskManager(self.position)
        self.adverse_selection_detector = AdverseSelectionDetector()
        
        # Performance tracking
        self.pnl_tracker = PnLTracker()
        self.quote_analytics = QuoteAnalytics()
        
        self.logger = ComponentLogger(f"MarketMaker_{asset}", "market_making")
    
    @abstractmethod
    def calculate_fair_value(self, market_data: MarketData) -> float:
        """Calculate fair value estimate"""
        pass
    
    @abstractmethod
    def calculate_optimal_spread(self, market_data: MarketData,
                               position: MarketMakerPosition) -> float:
        """Calculate optimal bid-ask spread"""
        pass
    
    @abstractmethod
    def calculate_inventory_skew(self, position: MarketMakerPosition,
                               market_data: MarketData) -> float:
        """Calculate inventory-based price skew"""
        pass
    
    def generate_quotes(self, market_data: MarketData) -> Optional[Quote]:
        """Generate bid/ask quotes"""
        
        # Check if we should quote
        if not self._should_quote(market_data):
            return None
        
        # Calculate fair value
        fair_value = self.calculate_fair_value(market_data)
        
        # Calculate optimal spread
        spread = self.calculate_optimal_spread(market_data, self.position)
        
        # Calculate skews
        inventory_skew = self.calculate_inventory_skew(self.position, market_data)
        flow_skew = self._calculate_flow_skew(market_data)
        
        # Generate quote parameters
        quote_params = QuoteParameters(
            fair_value=fair_value,
            volatility=self._estimate_volatility(market_data),
            spread=spread,
            inventory_skew=inventory_skew,
            flow_skew=flow_skew
        )
        
        # Create quote
        quote = self._create_quote(quote_params, market_data)
        
        # Risk checks
        if self._passes_risk_checks(quote, market_data):
            self.current_quotes[quote.quote_id] = quote
            self.quote_history.append(quote)
            return quote
        
        return None
    
    def handle_fill(self, fill: Fill) -> None:
        """Handle execution of our quote"""
        
        # Update position
        self._update_position(fill)
        
        # Update PnL
        self.pnl_tracker.add_fill(fill)
        
        # Check for adverse selection
        adverse_selection = self.adverse_selection_detector.analyze_fill(
            fill, self.current_quotes.get(fill.quote_id)
        )
        
        if adverse_selection.is_adverse:
            self._handle_adverse_selection(adverse_selection, fill)
        
        # Update quote analytics
        self.quote_analytics.record_fill(fill, adverse_selection)
        
        # Remove filled quote
        if fill.quote_id in self.current_quotes:
            del self.current_quotes[fill.quote_id]
    
    def _should_quote(self, market_data: MarketData) -> bool:
        """Determine if we should provide quotes"""
        
        # Check position limits
        if abs(self.position.inventory_ratio) > 0.9:
            return False
        
        # Check market conditions
        if market_data.volatility > self.config.get('max_volatility', 0.05):
            return False
        
        # Check spread conditions
        current_spread = market_data.ask_price - market_data.bid_price
        min_profitable_spread = self.config.get('min_spread', 0.01)
        
        if current_spread < min_profitable_spread:
            return False
        
        return True
    
    def _create_quote(self, params: QuoteParameters,
                     market_data: MarketData) -> Quote:
        """Create quote from parameters"""
        
        # Calculate bid/ask prices
        half_spread = params.spread / 2
        total_skew = params.inventory_skew + params.flow_skew
        
        bid_price = params.fair_value - half_spread + total_skew
        ask_price = params.fair_value + half_spread + total_skew
        
        # Apply price constraints
        bid_price = max(bid_price, market_data.bid_price * 0.99)  # Don't cross spread
        ask_price = min(ask_price, market_data.ask_price * 1.01)
        
        # Calculate sizes
        bid_size, ask_size = self._calculate_quote_sizes(params, market_data)
        
        quote = Quote(
            asset=self.asset,
            bid_price=round(bid_price, 4),
            ask_price=round(ask_price, 4),
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=datetime.now(),
            quote_id=self._generate_quote_id(),
            fair_value=params.fair_value,
            spread=params.spread,
            skew=total_skew
        )
        
        return quote
    
    def _calculate_quote_sizes(self, params: QuoteParameters,
                             market_data: MarketData) -> Tuple[int, int]:
        """Calculate bid/ask sizes based on position and risk"""
        
        base_size = int(params.base_size * params.size_multiplier)
        
        # Adjust based on inventory
        if self.position.is_long:
            # Reduce bid size, increase ask size when long
            bid_size = int(base_size * (1 - self.position.inventory_ratio))
            ask_size = int(base_size * (1 + self.position.inventory_ratio))
        elif self.position.is_short:
            # Increase bid size, reduce ask size when short
            bid_size = int(base_size * (1 + abs(self.position.inventory_ratio)))
            ask_size = int(base_size * (1 - abs(self.position.inventory_ratio)))
        else:
            # Balanced when flat
            bid_size = ask_size = base_size
        
        # Apply minimum sizes
        bid_size = max(bid_size, self.config.get('min_size', 10))
        ask_size = max(ask_size, self.config.get('min_size', 10))
        
        return bid_size, ask_size
```

### 2. Optimal Spread Calculation

```python
# src/market_making/spread_optimization.py
import scipy.optimize as opt
from scipy.stats import norm

class OptimalSpreadCalculator:
    """
    Calculates optimal bid-ask spread using theoretical models.
    Balances profit vs fill probability.
    """
    
    def __init__(self):
        self.logger = ComponentLogger("OptimalSpreadCalculator", "market_making")
    
    def calculate_avellaneda_stoikov_spread(self, 
                                          market_data: MarketData,
                                          position: MarketMakerPosition,
                                          risk_aversion: float = 0.1,
                                          time_horizon: float = 1.0) -> float:
        """
        Avellaneda-Stoikov optimal market making spread.
        Accounts for inventory risk and market impact.
        """
        
        # Market parameters
        volatility = self._estimate_volatility(market_data)
        arrival_rate = self._estimate_arrival_rate(market_data)
        
        # Current inventory
        q = position.quantity
        max_inventory = max(position.max_long, position.max_short)
        
        # Risk aversion parameter
        gamma = risk_aversion
        
        # Time to horizon
        T = time_horizon
        
        # Optimal spread (simplified Avellaneda-Stoikov)
        optimal_spread = (
            gamma * volatility**2 * T +
            (2/gamma) * np.log(1 + gamma/arrival_rate) +
            gamma * volatility**2 * abs(q) / max_inventory
        )
        
        return max(optimal_spread, 0.0001)  # Minimum 1 bps
    
    def calculate_ho_stoll_spread(self, market_data: MarketData,
                                inventory_cost: float = 0.001,
                                order_processing_cost: float = 0.0005) -> float:
        """
        Ho-Stoll spread model based on inventory and processing costs.
        """
        
        # Estimate adverse selection component
        adverse_selection = self._estimate_adverse_selection_cost(market_data)
        
        # Total spread
        total_spread = 2 * (
            inventory_cost +
            order_processing_cost +
            adverse_selection
        )
        
        return total_spread
    
    def calculate_dynamic_spread(self, market_data: MarketData,
                               position: MarketMakerPosition,
                               recent_fills: List[Fill]) -> float:
        """
        Dynamic spread based on recent market activity.
        """
        
        # Base spread from volatility
        volatility = self._estimate_volatility(market_data)
        base_spread = 2 * volatility / np.sqrt(252)  # Daily vol to spread
        
        # Adjust for recent adverse selection
        if recent_fills:
            adverse_ratio = self._calculate_adverse_fill_ratio(recent_fills)
            adverse_adjustment = 1 + adverse_ratio
        else:
            adverse_adjustment = 1.0
        
        # Adjust for inventory
        inventory_adjustment = 1 + abs(position.inventory_ratio) * 0.5
        
        # Adjust for market activity
        activity_adjustment = self._calculate_activity_adjustment(market_data)
        
        dynamic_spread = (
            base_spread * 
            adverse_adjustment * 
            inventory_adjustment * 
            activity_adjustment
        )
        
        return dynamic_spread
    
    def _estimate_arrival_rate(self, market_data: MarketData) -> float:
        """Estimate order arrival rate"""
        # Use recent volume as proxy for arrival rate
        recent_volume = market_data.volume_1min
        avg_trade_size = 100  # Assume average trade size
        
        arrival_rate = recent_volume / avg_trade_size / 60  # Per second
        
        return max(arrival_rate, 0.1)  # Minimum rate
    
    def _estimate_adverse_selection_cost(self, market_data: MarketData) -> float:
        """Estimate cost of adverse selection"""
        
        # Use price impact as proxy
        # Higher impact means more informed trading
        
        volume = market_data.volume_1min
        price_change = abs(market_data.close - market_data.open)
        
        if volume > 0:
            impact = price_change / (volume / 1000000)  # Per million shares
            adverse_cost = min(impact * 0.5, 0.005)  # Cap at 50bps
        else:
            adverse_cost = 0.001  # Default 10bps
        
        return adverse_cost

class InventoryRiskManager:
    """
    Manages inventory risk for market maker.
    Calculates hedging needs and risk metrics.
    """
    
    def __init__(self, position: MarketMakerPosition):
        self.position = position
        self.hedge_thresholds = {
            'warning': 0.7,
            'critical': 0.9
        }
        self.logger = ComponentLogger("InventoryRiskManager", "market_making")
    
    def calculate_inventory_risk(self, market_data: MarketData) -> float:
        """Calculate current inventory risk in dollars"""
        
        # Price risk from inventory
        volatility = self._estimate_daily_volatility(market_data)
        position_value = abs(self.position.quantity) * market_data.mid_price
        
        # 1-day 95% VaR
        inventory_var = position_value * volatility * 1.645
        
        return inventory_var
    
    def should_hedge(self, market_data: MarketData) -> Tuple[bool, str]:
        """Determine if position should be hedged"""
        
        inventory_ratio = abs(self.position.inventory_ratio)
        
        if inventory_ratio > self.hedge_thresholds['critical']:
            return True, "critical_threshold"
        
        # Risk-based hedging
        inventory_risk = self.calculate_inventory_risk(market_data)
        position_value = abs(self.position.quantity) * market_data.mid_price
        
        if position_value > 0:
            risk_ratio = inventory_risk / position_value
            
            if risk_ratio > 0.05:  # 5% daily risk
                return True, "risk_threshold"
        
        return False, "no_hedge_needed"
    
    def calculate_hedge_size(self, market_data: MarketData) -> int:
        """Calculate optimal hedge size"""
        
        # Target inventory level
        if self.position.is_long:
            target_ratio = 0.3  # Reduce to 30% of max
            target_quantity = self.position.max_long * target_ratio
            hedge_size = -(self.position.quantity - target_quantity)
        else:
            target_ratio = 0.3
            target_quantity = -self.position.max_short * target_ratio
            hedge_size = -(self.position.quantity - target_quantity)
        
        return int(hedge_size)
    
    def execute_hedge(self, hedge_size: int,
                     execution_handler: ExecutionHandler) -> Optional[Order]:
        """Execute hedge trade"""
        
        if abs(hedge_size) < 10:  # Minimum hedge size
            return None
        
        hedge_order = Order(
            asset=self.position.asset,
            quantity=abs(hedge_size),
            side=OrderSide.SELL if hedge_size < 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            metadata={
                'trade_type': 'hedge',
                'reason': 'inventory_management',
                'original_position': self.position.quantity
            }
        )
        
        self.logger.info(f"Executing hedge: {hedge_size} shares")
        
        return execution_handler.send_order(hedge_order)
```

### 3. Adverse Selection Detection

```python
# src/market_making/adverse_selection.py
@dataclass
class AdverseSelectionSignal:
    """Signal indicating adverse selection"""
    is_adverse: bool
    confidence: float
    signal_type: str
    
    # Metrics
    immediate_impact: float
    delayed_impact: float
    flow_imbalance: float
    
    # Recommendations
    spread_adjustment: float
    size_adjustment: float

class AdverseSelectionDetector:
    """
    Detects when market maker is being picked off by informed traders.
    Uses multiple signals to identify adverse selection.
    """
    
    def __init__(self):
        self.fill_history: List[Fill] = []
        self.impact_tracker = ImpactTracker()
        self.flow_analyzer = OrderFlowAnalyzer()
        self.logger = ComponentLogger("AdverseSelectionDetector", "market_making")
    
    def analyze_fill(self, fill: Fill, quote: Optional[Quote]) -> AdverseSelectionSignal:
        """Analyze if a fill shows adverse selection"""
        
        # Store fill
        self.fill_history.append(fill)
        
        # Multiple adverse selection tests
        signals = []
        
        # Test 1: Immediate price impact
        immediate_signal = self._test_immediate_impact(fill)
        signals.append(immediate_signal)
        
        # Test 2: Delayed price impact
        if len(self.fill_history) > 10:
            delayed_signal = self._test_delayed_impact(fill)
            signals.append(delayed_signal)
        
        # Test 3: Order flow imbalance
        flow_signal = self._test_flow_imbalance(fill)
        signals.append(flow_signal)
        
        # Test 4: Fill size relative to quote
        if quote:
            size_signal = self._test_fill_size_pattern(fill, quote)
            signals.append(size_signal)
        
        # Test 5: Timing patterns
        timing_signal = self._test_timing_patterns(fill)
        signals.append(timing_signal)
        
        # Aggregate signals
        return self._aggregate_signals(signals, fill)
    
    def _test_immediate_impact(self, fill: Fill) -> Dict:
        """Test for immediate price impact after fill"""
        
        # Get price movement in next few seconds
        post_fill_impact = self.impact_tracker.get_price_impact(
            fill.timestamp,
            window_seconds=30
        )
        
        # Adverse if price moved significantly against us
        if fill.side == OrderSide.BUY:  # We sold
            # Adverse if price continued down
            is_adverse = post_fill_impact < -0.0002  # -2 bps
        else:  # We bought
            # Adverse if price continued up
            is_adverse = post_fill_impact > 0.0002  # +2 bps
        
        return {
            'type': 'immediate_impact',
            'is_adverse': is_adverse,
            'impact': abs(post_fill_impact),
            'confidence': min(abs(post_fill_impact) * 10000, 1.0)  # Confidence from impact size
        }
    
    def _test_delayed_impact(self, fill: Fill) -> Dict:
        """Test for delayed price impact (5-15 minutes)"""
        
        # Get price movement over longer horizon
        delayed_impact = self.impact_tracker.get_price_impact(
            fill.timestamp,
            window_seconds=600  # 10 minutes
        )
        
        # Similar logic but lower threshold
        if fill.side == OrderSide.BUY:
            is_adverse = delayed_impact < -0.0005  # -5 bps
        else:
            is_adverse = delayed_impact > 0.0005
        
        return {
            'type': 'delayed_impact', 
            'is_adverse': is_adverse,
            'impact': abs(delayed_impact),
            'confidence': min(abs(delayed_impact) * 5000, 1.0)
        }
    
    def _test_flow_imbalance(self, fill: Fill) -> Dict:
        """Test for order flow imbalance"""
        
        # Analyze recent order flow
        flow_imbalance = self.flow_analyzer.calculate_flow_imbalance(
            window_minutes=5
        )
        
        # Adverse if we're consistently being hit on one side
        if fill.side == OrderSide.BUY and flow_imbalance > 0.7:
            # We bought but flow is very buy-heavy (we're picking up unwanted inventory)
            is_adverse = True
            confidence = flow_imbalance
        elif fill.side == OrderSide.SELL and flow_imbalance < -0.7:
            # We sold but flow is very sell-heavy
            is_adverse = True
            confidence = abs(flow_imbalance)
        else:
            is_adverse = False
            confidence = 0.0
        
        return {
            'type': 'flow_imbalance',
            'is_adverse': is_adverse,
            'imbalance': flow_imbalance,
            'confidence': confidence
        }
    
    def _test_fill_size_pattern(self, fill: Fill, quote: Quote) -> Dict:
        """Test if fill size suggests informed trading"""
        
        # Large fills relative to quote size can indicate informed trading
        fill_ratio = fill.quantity / (quote.bid_size if fill.side == OrderSide.SELL else quote.ask_size)
        
        # Very large fills more likely to be informed
        is_adverse = fill_ratio > 0.8  # Filled 80%+ of quoted size
        confidence = min(fill_ratio, 1.0)
        
        return {
            'type': 'fill_size',
            'is_adverse': is_adverse,
            'fill_ratio': fill_ratio,
            'confidence': confidence if is_adverse else 0.0
        }
    
    def _test_timing_patterns(self, fill: Fill) -> Dict:
        """Test for suspicious timing patterns"""
        
        # Check if fills are clustered in time (suggests coordinated informed trading)
        recent_fills = [
            f for f in self.fill_history[-10:]
            if (fill.timestamp - f.timestamp).total_seconds() < 60
        ]
        
        fill_density = len(recent_fills)
        
        # High density of fills suggests potential informed trading
        is_adverse = fill_density > 5  # More than 5 fills in 1 minute
        confidence = min(fill_density / 10, 1.0)
        
        return {
            'type': 'timing_pattern',
            'is_adverse': is_adverse,
            'fill_density': fill_density,
            'confidence': confidence if is_adverse else 0.0
        }
    
    def _aggregate_signals(self, signals: List[Dict], fill: Fill) -> AdverseSelectionSignal:
        """Aggregate multiple adverse selection signals"""
        
        # Count adverse signals
        adverse_count = sum(1 for s in signals if s['is_adverse'])
        total_signals = len(signals)
        
        # Weighted confidence
        total_confidence = sum(s['confidence'] for s in signals)
        avg_confidence = total_confidence / total_signals if total_signals > 0 else 0
        
        # Overall assessment
        is_adverse = adverse_count >= 2 or avg_confidence > 0.7
        
        # Extract specific metrics
        immediate_impact = next((s['impact'] for s in signals if s['type'] == 'immediate_impact'), 0)
        delayed_impact = next((s['impact'] for s in signals if s['type'] == 'delayed_impact'), 0)
        flow_imbalance = next((s.get('imbalance', 0) for s in signals if s['type'] == 'flow_imbalance'), 0)
        
        # Calculate adjustments
        spread_adjustment = self._calculate_spread_adjustment(avg_confidence)
        size_adjustment = self._calculate_size_adjustment(avg_confidence)
        
        return AdverseSelectionSignal(
            is_adverse=is_adverse,
            confidence=avg_confidence,
            signal_type=f"{adverse_count}/{total_signals}_adverse",
            immediate_impact=immediate_impact,
            delayed_impact=delayed_impact,
            flow_imbalance=flow_imbalance,
            spread_adjustment=spread_adjustment,
            size_adjustment=size_adjustment
        )
    
    def _calculate_spread_adjustment(self, confidence: float) -> float:
        """Calculate how much to widen spread due to adverse selection"""
        # Widen spread by up to 50% based on confidence
        return 1 + (confidence * 0.5)
    
    def _calculate_size_adjustment(self, confidence: float) -> float:
        """Calculate how much to reduce quote size due to adverse selection"""
        # Reduce size by up to 30% based on confidence
        return 1 - (confidence * 0.3)

class ImpactTracker:
    """Tracks price impact of trades"""
    
    def __init__(self):
        self.price_history: List[Tuple[datetime, float]] = []
    
    def add_price(self, timestamp: datetime, price: float) -> None:
        """Add price observation"""
        self.price_history.append((timestamp, price))
        
        # Keep only recent history
        cutoff = timestamp - timedelta(hours=1)
        self.price_history = [
            (ts, px) for ts, px in self.price_history 
            if ts > cutoff
        ]
    
    def get_price_impact(self, reference_time: datetime,
                        window_seconds: int = 30) -> float:
        """Get price change after reference time"""
        
        # Find price at reference time
        ref_price = self._get_price_at_time(reference_time)
        if ref_price is None:
            return 0.0
        
        # Find price after window
        end_time = reference_time + timedelta(seconds=window_seconds)
        end_price = self._get_price_at_time(end_time)
        if end_price is None:
            return 0.0
        
        # Calculate impact
        return (end_price - ref_price) / ref_price
    
    def _get_price_at_time(self, target_time: datetime) -> Optional[float]:
        """Get price closest to target time"""
        if not self.price_history:
            return None
        
        # Find closest price
        closest = min(
            self.price_history,
            key=lambda x: abs((x[0] - target_time).total_seconds())
        )
        
        # Only use if within 30 seconds
        if abs((closest[0] - target_time).total_seconds()) < 30:
            return closest[1]
        
        return None
```

### 4. Multi-Level Order Book Strategy

```python
# src/market_making/multi_level.py
class MultiLevelMarketMaker(MarketMaker):
    """
    Advanced market maker with multiple price levels.
    Provides liquidity at different price points.
    """
    
    def __init__(self, asset: str, config: Dict):
        super().__init__(asset, config)
        
        self.n_levels = config.get('n_levels', 3)
        self.level_spacing = config.get('level_spacing', 0.0005)  # 5 bps between levels
        self.size_decay = config.get('size_decay', 0.7)  # Size decay factor per level
        
        # Track quotes by level
        self.quotes_by_level: Dict[int, Quote] = {}
    
    def generate_quotes(self, market_data: MarketData) -> List[Quote]:
        """Generate quotes for multiple levels"""
        
        if not self._should_quote(market_data):
            return []
        
        # Calculate base parameters
        fair_value = self.calculate_fair_value(market_data)
        base_spread = self.calculate_optimal_spread(market_data, self.position)
        inventory_skew = self.calculate_inventory_skew(self.position, market_data)
        
        quotes = []
        
        for level in range(self.n_levels):
            # Calculate level-specific parameters
            level_spread = base_spread * (1 + level * 0.2)  # Wider spreads further out
            level_spacing = self.level_spacing * level
            level_size_multiplier = self.size_decay ** level
            
            # Create level quote
            quote_params = QuoteParameters(
                fair_value=fair_value,
                volatility=self._estimate_volatility(market_data),
                spread=level_spread,
                inventory_skew=inventory_skew * (1 + level * 0.1),  # More skew further out
                base_size=self.config.get('base_size', 100),
                size_multiplier=level_size_multiplier
            )
            
            # Adjust for level spacing
            if level > 0:
                if self.position.is_long:
                    # When long, space ask levels wider
                    quote_params.fair_value += level_spacing
                elif self.position.is_short:
                    # When short, space bid levels wider
                    quote_params.fair_value -= level_spacing
            
            quote = self._create_quote(quote_params, market_data)
            quote.metadata = {'level': level}
            
            if self._passes_risk_checks(quote, market_data):
                quotes.append(quote)
                self.quotes_by_level[level] = quote
        
        return quotes
    
    def handle_fill(self, fill: Fill) -> None:
        """Handle fill and adjust remaining levels"""
        super().handle_fill(fill)
        
        # Determine which level was hit
        filled_level = fill.metadata.get('level', 0)
        
        # Cancel quotes at same or more aggressive levels
        levels_to_cancel = []
        
        if fill.side == OrderSide.BUY:  # Our ask was hit
            # Cancel more aggressive ask levels
            for level in range(filled_level + 1):
                if level in self.quotes_by_level:
                    levels_to_cancel.append(level)
        else:  # Our bid was hit
            # Cancel more aggressive bid levels  
            for level in range(filled_level + 1):
                if level in self.quotes_by_level:
                    levels_to_cancel.append(level)
        
        # Remove cancelled quotes
        for level in levels_to_cancel:
            if level in self.quotes_by_level:
                del self.quotes_by_level[level]
        
        # Refresh quotes after brief delay
        self._schedule_quote_refresh(delay_ms=100)

class CrossVenueArbitrageDetector:
    """
    Detects arbitrage opportunities across venues.
    Enables market maker to capture cross-venue spreads.
    """
    
    def __init__(self, venues: List[str]):
        self.venues = venues
        self.venue_data: Dict[str, MarketData] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.logger = ComponentLogger("CrossVenueArbitrage", "market_making")
    
    def update_venue_data(self, venue: str, market_data: MarketData) -> None:
        """Update market data for a venue"""
        self.venue_data[venue] = market_data
        
        # Check for new arbitrage opportunities
        opportunities = self._scan_arbitrage_opportunities()
        
        for opp in opportunities:
            if self._is_profitable_arbitrage(opp):
                self.arbitrage_opportunities.append(opp)
                self.logger.info(f"Arbitrage opportunity: {opp}")
    
    def _scan_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage across all venue pairs"""
        opportunities = []
        
        for venue1 in self.venues:
            for venue2 in self.venues:
                if venue1 >= venue2:  # Avoid duplicates
                    continue
                
                if venue1 not in self.venue_data or venue2 not in self.venue_data:
                    continue
                
                data1 = self.venue_data[venue1]
                data2 = self.venue_data[venue2]
                
                # Check both directions
                # Direction 1: Buy on venue1, sell on venue2
                if data1.ask_price < data2.bid_price:
                    profit = data2.bid_price - data1.ask_price
                    size = min(data1.ask_size, data2.bid_size)
                    
                    opportunities.append(ArbitrageOpportunity(
                        buy_venue=venue1,
                        sell_venue=venue2,
                        buy_price=data1.ask_price,
                        sell_price=data2.bid_price,
                        size=size,
                        profit_per_share=profit,
                        total_profit=profit * size
                    ))
                
                # Direction 2: Buy on venue2, sell on venue1
                if data2.ask_price < data1.bid_price:
                    profit = data1.bid_price - data2.ask_price
                    size = min(data2.ask_size, data1.bid_size)
                    
                    opportunities.append(ArbitrageOpportunity(
                        buy_venue=venue2,
                        sell_venue=venue1,
                        buy_price=data2.ask_price,
                        sell_price=data1.bid_price,
                        size=size,
                        profit_per_share=profit,
                        total_profit=profit * size
                    ))
        
        return opportunities
    
    def _is_profitable_arbitrage(self, opp: ArbitrageOpportunity) -> bool:
        """Check if arbitrage is profitable after costs"""
        
        # Estimate transaction costs
        buy_cost = opp.buy_price * opp.size * 0.0005  # 5 bps
        sell_cost = opp.sell_price * opp.size * 0.0005
        total_costs = buy_cost + sell_cost
        
        # Check if profit exceeds costs
        net_profit = opp.total_profit - total_costs
        
        return net_profit > 10  # Minimum $10 profit
```

### 5. Performance Analytics

```python
# src/market_making/analytics.py
class MarketMakingAnalytics:
    """
    Comprehensive analytics for market making performance.
    Tracks profitability, risk, and operational metrics.
    """
    
    def __init__(self):
        self.trades: List[Fill] = []
        self.quotes: List[Quote] = []
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.inventory_history: List[Tuple[datetime, float]] = []
        
    def add_trade(self, fill: Fill) -> None:
        """Add trade to analytics"""
        self.trades.append(fill)
    
    def add_quote(self, quote: Quote) -> None:
        """Add quote to analytics"""
        self.quotes.append(quote)
    
    def calculate_performance_metrics(self, time_period: timedelta) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        cutoff_time = datetime.now() - time_period
        
        # Filter data to time period
        period_trades = [t for t in self.trades if t.timestamp > cutoff_time]
        period_quotes = [q for q in self.quotes if q.timestamp > cutoff_time]
        
        if not period_trades:
            return {}
        
        # Basic metrics
        total_pnl = sum(t.pnl for t in period_trades)
        total_volume = sum(t.quantity * t.price for t in period_trades)
        n_trades = len(period_trades)
        
        # Profitability metrics
        avg_pnl_per_trade = total_pnl / n_trades if n_trades > 0 else 0
        pnl_per_share = total_pnl / sum(t.quantity for t in period_trades) if period_trades else 0
        
        # Quote metrics
        avg_spread = np.mean([q.spread_bps for q in period_quotes]) if period_quotes else 0
        fill_rate = len(period_trades) / len(period_quotes) if period_quotes else 0
        
        # Risk metrics
        inventory_std = self._calculate_inventory_volatility(cutoff_time)
        max_inventory = max(abs(pos) for _, pos in self.inventory_history 
                          if _ > cutoff_time) if self.inventory_history else 0
        
        # Adverse selection metrics
        adverse_selection_rate = self._calculate_adverse_selection_rate(period_trades)
        
        return {
            'total_pnl': total_pnl,
            'total_volume': total_volume,
            'n_trades': n_trades,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'pnl_per_share': pnl_per_share,
            'avg_spread_bps': avg_spread,
            'fill_rate': fill_rate,
            'inventory_volatility': inventory_std,
            'max_inventory': max_inventory,
            'adverse_selection_rate': adverse_selection_rate,
            'sharpe_ratio': self._calculate_sharpe_ratio(cutoff_time),
            'profit_factor': self._calculate_profit_factor(period_trades)
        }
    
    def _calculate_adverse_selection_rate(self, trades: List[Fill]) -> float:
        """Calculate rate of adverse selection"""
        if not trades:
            return 0.0
        
        adverse_trades = sum(1 for t in trades 
                           if t.metadata.get('adverse_selection', False))
        
        return adverse_trades / len(trades)
    
    def _calculate_sharpe_ratio(self, cutoff_time: datetime) -> float:
        """Calculate Sharpe ratio of PnL"""
        period_pnl = [(ts, pnl) for ts, pnl in self.pnl_history if ts > cutoff_time]
        
        if len(period_pnl) < 2:
            return 0.0
        
        returns = [pnl for _, pnl in period_pnl]
        
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        # Daily metrics
        daily_metrics = self.calculate_performance_metrics(timedelta(days=1))
        
        # Weekly metrics  
        weekly_metrics = self.calculate_performance_metrics(timedelta(days=7))
        
        # Monthly metrics
        monthly_metrics = self.calculate_performance_metrics(timedelta(days=30))
        
        report = f"""
        Market Making Performance Report
        ===============================
        
        Daily Performance:
        - Total PnL: ${daily_metrics.get('total_pnl', 0):.2f}
        - Trades: {daily_metrics.get('n_trades', 0)}
        - Avg Spread: {daily_metrics.get('avg_spread_bps', 0):.1f} bps
        - Fill Rate: {daily_metrics.get('fill_rate', 0):.1%}
        - Adverse Selection: {daily_metrics.get('adverse_selection_rate', 0):.1%}
        
        Weekly Performance:
        - Total PnL: ${weekly_metrics.get('total_pnl', 0):.2f}
        - Sharpe Ratio: {weekly_metrics.get('sharpe_ratio', 0):.2f}
        - Max Inventory: {weekly_metrics.get('max_inventory', 0):.0f}
        
        Monthly Performance:
        - Total PnL: ${monthly_metrics.get('total_pnl', 0):.2f}
        - Total Volume: ${monthly_metrics.get('total_volume', 0):,.0f}
        - PnL per Share: ${monthly_metrics.get('pnl_per_share', 0):.4f}
        """
        
        return report
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_4_market_making.py`:

```python
class TestMarketMaker:
    """Test market making core functionality"""
    
    def test_quote_generation(self):
        """Test quote generation logic"""
        config = {
            'max_long': 10000,
            'max_short': 10000,
            'base_size': 100
        }
        
        mm = TestMarketMaker('SPY', config)
        market_data = create_test_market_data(
            bid=400.00, ask=400.05, mid=400.025
        )
        
        quote = mm.generate_quotes(market_data)
        
        assert quote is not None
        assert quote.bid_price < quote.ask_price
        assert quote.spread_bps > 0
        assert quote.bid_size > 0
        assert quote.ask_size > 0
    
    def test_inventory_skew(self):
        """Test inventory-based price skew"""
        mm = TestMarketMaker('SPY', {})
        
        # Long position should skew quotes down
        mm.position.quantity = 5000
        mm.position.max_long = 10000
        
        market_data = create_test_market_data()
        skew = mm.calculate_inventory_skew(mm.position, market_data)
        
        assert skew < 0  # Negative skew when long

class TestOptimalSpread:
    """Test spread optimization"""
    
    def test_avellaneda_stoikov_spread(self):
        """Test A-S optimal spread calculation"""
        calculator = OptimalSpreadCalculator()
        
        market_data = create_test_market_data(volatility=0.02)
        position = create_test_position(quantity=1000)
        
        spread = calculator.calculate_avellaneda_stoikov_spread(
            market_data, position, risk_aversion=0.1
        )
        
        assert spread > 0
        assert spread < 0.01  # Reasonable spread
    
    def test_spread_increases_with_inventory(self):
        """Test that spread increases with inventory"""
        calculator = OptimalSpreadCalculator()
        market_data = create_test_market_data()
        
        # Small position
        small_position = create_test_position(quantity=100)
        small_spread = calculator.calculate_avellaneda_stoikov_spread(
            market_data, small_position
        )
        
        # Large position
        large_position = create_test_position(quantity=5000)
        large_spread = calculator.calculate_avellaneda_stoikov_spread(
            market_data, large_position
        )
        
        assert large_spread > small_spread

class TestAdverseSelection:
    """Test adverse selection detection"""
    
    def test_immediate_impact_detection(self):
        """Test immediate price impact detection"""
        detector = AdverseSelectionDetector()
        
        # Mock impact tracker
        detector.impact_tracker.get_price_impact = lambda ts, window: -0.0005  # -5 bps
        
        fill = create_test_fill(side=OrderSide.BUY)  # We sold
        
        signal = detector.analyze_fill(fill, None)
        
        # Should detect adverse selection (price moved against us)
        assert signal.is_adverse
        assert signal.confidence > 0
```

### Integration Tests

Create `tests/integration/test_step10_4_market_making_integration.py`:

```python
def test_complete_market_making_cycle():
    """Test full market making workflow"""
    # Setup market maker
    config = {
        'max_long': 10000,
        'max_short': 10000,
        'base_size': 100,
        'min_spread': 0.0005
    }
    
    mm = MultiLevelMarketMaker('SPY', config)
    
    # Market data stream
    market_stream = create_test_market_stream()
    
    # Execution handler
    exec_handler = SimulatedExecutionHandler(
        fill_probability=0.1,  # 10% chance per quote
        adverse_selection_rate=0.05  # 5% adverse fills
    )
    
    # Run market making for period
    total_pnl = 0
    quotes_sent = 0
    fills_received = 0
    
    for i in range(1000):  # 1000 cycles
        market_data = market_stream.get_next()
        
        # Generate quotes
        quotes = mm.generate_quotes(market_data)
        
        for quote in quotes:
            quotes_sent += 1
            
            # Send to market
            fill = exec_handler.try_fill(quote, market_data)
            
            if fill:
                fills_received += 1
                mm.handle_fill(fill)
                total_pnl += fill.pnl
                
                # Check for hedging need
                should_hedge, reason = mm.risk_manager.should_hedge(market_data)
                if should_hedge:
                    hedge_size = mm.risk_manager.calculate_hedge_size(market_data)
                    hedge_order = mm.risk_manager.execute_hedge(hedge_size, exec_handler)
    
    # Verify results
    assert quotes_sent > 0
    assert fills_received > 0
    fill_rate = fills_received / quotes_sent
    assert 0.05 < fill_rate < 0.2  # Reasonable fill rate
    
    # Should be profitable on average
    avg_pnl_per_fill = total_pnl / fills_received if fills_received > 0 else 0
    assert avg_pnl_per_fill > 0  # Positive expected value

def test_multi_venue_arbitrage():
    """Test cross-venue arbitrage detection"""
    venues = ['NYSE', 'NASDAQ', 'ARCA']
    arbitrage_detector = CrossVenueArbitrageDetector(venues)
    
    # Setup price differential
    arbitrage_detector.update_venue_data('NYSE', MarketData(
        bid=400.00, ask=400.05, bid_size=1000, ask_size=1000
    ))
    
    arbitrage_detector.update_venue_data('NASDAQ', MarketData(
        bid=400.06, ask=400.11, bid_size=500, ask_size=500  # Higher prices
    ))
    
    # Should detect arbitrage opportunity
    opportunities = arbitrage_detector.arbitrage_opportunities
    assert len(opportunities) > 0
    
    # Verify opportunity details
    opp = opportunities[0]
    assert opp.buy_venue == 'NYSE'
    assert opp.sell_venue == 'NASDAQ'
    assert opp.profit_per_share > 0
```

### System Tests

Create `tests/system/test_step10_4_production_market_making.py`:

```python
def test_high_frequency_market_making():
    """Test market making at high frequency"""
    mm = ProductionMarketMaker('SPY', production_config)
    
    # High-frequency market data (updates every 100ms)
    market_stream = HighFrequencyMarketStream('SPY', update_frequency_ms=100)
    
    execution_stats = {
        'quotes_per_second': [],
        'response_times': [],
        'memory_usage': []
    }
    
    start_time = time.time()
    
    for i in range(10000):  # 10,000 updates (~16 minutes at 100ms)
        cycle_start = time.time()
        
        # Get market update
        market_data = market_stream.get_next()
        
        # Generate quotes
        quotes = mm.generate_quotes(market_data)
        
        # Process any fills
        for fill in market_stream.get_fills():
            mm.handle_fill(fill)
        
        cycle_time = time.time() - cycle_start
        execution_stats['response_times'].append(cycle_time)
        
        # Track performance every second
        if i % 10 == 0:  # Every 1 second
            qps = len(quotes) / (cycle_time + 0.001)  # Avoid division by zero
            execution_stats['quotes_per_second'].append(qps)
            
            memory_mb = get_memory_usage_mb()
            execution_stats['memory_usage'].append(memory_mb)
    
    total_time = time.time() - start_time
    
    # Performance requirements
    avg_response_time = np.mean(execution_stats['response_times'])
    p99_response_time = np.percentile(execution_stats['response_times'], 99)
    avg_memory = np.mean(execution_stats['memory_usage'])
    
    assert avg_response_time < 0.010  # 10ms average
    assert p99_response_time < 0.050  # 50ms p99
    assert avg_memory < 500  # 500MB max memory
    
    # Market making performance
    analytics = mm.get_analytics()
    performance = analytics.calculate_performance_metrics(timedelta(seconds=total_time))
    
    assert performance['avg_spread_bps'] > 1  # At least 1 bps spread
    assert performance['fill_rate'] > 0.01  # At least 1% fill rate
    assert performance['total_pnl'] > -1000  # Max $1000 loss

def test_stress_testing():
    """Test market making under stress conditions"""
    stress_scenarios = [
        'high_volatility',
        'low_liquidity', 
        'one_sided_flow',
        'flash_crash',
        'news_event'
    ]
    
    mm = StressTestMarketMaker('SPY', stress_test_config)
    
    results = {}
    
    for scenario in stress_scenarios:
        # Generate stress scenario
        stress_data = generate_stress_scenario(scenario, duration_minutes=10)
        
        # Reset market maker state
        mm.reset()
        
        scenario_stats = {
            'max_inventory': 0,
            'max_loss': 0,
            'quotes_cancelled': 0,
            'adverse_fills': 0
        }
        
        # Run through scenario
        for market_data in stress_data:
            quotes = mm.generate_quotes(market_data)
            
            # Process fills
            for fill in market_data.fills:
                mm.handle_fill(fill)
                
                # Track adverse selection
                if fill.metadata.get('adverse_selection'):
                    scenario_stats['adverse_fills'] += 1
            
            # Track inventory
            current_inventory = abs(mm.position.quantity)
            scenario_stats['max_inventory'] = max(
                scenario_stats['max_inventory'], 
                current_inventory
            )
            
            # Track losses
            if mm.position.unrealized_pnl < scenario_stats['max_loss']:
                scenario_stats['max_loss'] = mm.position.unrealized_pnl
        
        results[scenario] = scenario_stats
        
        # Verify risk controls worked
        assert scenario_stats['max_inventory'] < mm.position.max_long * 1.1  # Within 110% of limit
        assert scenario_stats['max_loss'] > -10000  # Max $10k loss per scenario
    
    # Overall stress test pass
    total_adverse = sum(s['adverse_fills'] for s in results.values())
    total_fills = sum(len(stress_data) for stress_data in stress_scenarios)
    
    adverse_rate = total_adverse / total_fills if total_fills > 0 else 0
    assert adverse_rate < 0.3  # Less than 30% adverse selection under stress
```

## ✅ Validation Checklist

### Core Framework
- [ ] Quote generation working
- [ ] Position tracking accurate
- [ ] Risk limits enforced
- [ ] PnL calculation correct

### Spread Optimization
- [ ] A-S model implemented
- [ ] Dynamic adjustments working
- [ ] Inventory impact included
- [ ] Market conditions considered

### Adverse Selection
- [ ] Multiple detection methods
- [ ] Real-time analysis working
- [ ] Spread adjustments automatic
- [ ] Size reductions applied

### Multi-Level Strategy
- [ ] Multiple quote levels
- [ ] Level spacing optimal
- [ ] Size decay applied
- [ ] Level cancellation logic

### Cross-Venue Features
- [ ] Arbitrage detection working
- [ ] Venue routing optimal
- [ ] Latency considerations
- [ ] Cost calculations accurate

## 📊 Performance Benchmarks

### Real-time Performance
- Quote generation: < 5ms
- Fill processing: < 2ms
- Risk checks: < 1ms
- Adverse selection analysis: < 10ms

### Trading Performance
- Average spread: 1-5 bps
- Fill rate: 5-15%
- Adverse selection rate: < 10%
- Daily Sharpe ratio: > 2.0

### Risk Management
- Inventory within limits: 100%
- Max intraday loss: < 0.5% of capital
- Hedge execution: < 30 seconds
- Position duration: < 4 hours

## 🐛 Common Issues

1. **Adverse Selection**
   - Monitor fill patterns
   - Adjust spreads dynamically
   - Implement size limits
   - Use multiple detection methods

2. **Inventory Risk**
   - Set strict limits
   - Hedge proactively
   - Monitor correlations
   - Track time in position

3. **Technology Latency**
   - Optimize code paths
   - Use efficient data structures
   - Pre-calculate when possible
   - Monitor system performance

## 🎯 Success Criteria

Step 10.4 is complete when:
1. ✅ All market making components implemented
2. ✅ Optimal spread calculation working
3. ✅ Adverse selection detection functional
4. ✅ Multi-level strategies operational
5. ✅ Performance benchmarks met

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.5: Regime Adaptation](step-10.5-regime-adaptation.md)

## 📚 Additional Resources

- [Market Making Theory](../references/market-making-theory.md)
- [Optimal Market Making Models](../references/optimal-mm-models.md)
- [Inventory Risk Management](../references/inventory-risk-management.md)
- [High-Frequency Trading Systems](../references/hft-systems.md)