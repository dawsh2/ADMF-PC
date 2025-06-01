# Step 10.3: Execution Algorithms

**Status**: Intermediate Complexity Step
**Complexity**: High
**Prerequisites**: [Step 10.2: Multi-Asset Support](step-10.2-multi-asset.md) completed
**Architecture Ref**: [Execution Architecture](../architecture/execution-architecture.md)

## 🎯 Objective

Implement sophisticated execution algorithms:
- TWAP (Time-Weighted Average Price) implementation
- VWAP (Volume-Weighted Average Price) execution
- Implementation shortfall minimization
- Iceberg and hidden order strategies
- Smart order routing across venues
- Adaptive execution based on market conditions

## 📋 Required Reading

Before starting:
1. [Algorithmic Trading Concepts](../references/algo-trading-concepts.md)
2. [Market Microstructure for Execution](../references/execution-microstructure.md)
3. [Transaction Cost Analysis](../references/tca-guide.md)

## 🏗️ Implementation Tasks

### 1. Core Execution Framework

```python
# src/execution/algorithms/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class ExecutionStyle(Enum):
    """Execution algorithm styles"""
    AGGRESSIVE = "aggressive"  # Prioritize speed
    PASSIVE = "passive"       # Prioritize price
    NORMAL = "normal"         # Balanced approach
    ADAPTIVE = "adaptive"     # Adjust based on conditions

@dataclass
class ExecutionPlan:
    """Execution plan for an order"""
    parent_order: Order
    child_orders: List[Order]
    schedule: pd.DataFrame  # Time-based schedule
    
    # Execution parameters
    start_time: datetime
    end_time: datetime
    style: ExecutionStyle
    
    # Constraints
    min_fill_size: float
    max_participation_rate: float
    price_limit: Optional[float]
    
    # Tracking
    executed_quantity: float = 0
    remaining_quantity: float = 0
    average_price: float = 0
    slippage: float = 0
    
    def update_progress(self, fill: Fill) -> None:
        """Update execution progress with new fill"""
        # Update quantities
        old_total = self.executed_quantity * self.average_price
        self.executed_quantity += fill.quantity
        
        # Update average price
        if self.executed_quantity > 0:
            self.average_price = (old_total + fill.quantity * fill.price) / self.executed_quantity
        
        self.remaining_quantity = self.parent_order.quantity - self.executed_quantity

class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.completed_plans: List[ExecutionPlan] = []
        self.logger = ComponentLogger(f"ExecutionAlgo_{name}", "execution")
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'completed_orders': 0,
            'avg_slippage': 0,
            'avg_participation': 0
        }
    
    @abstractmethod
    def create_execution_plan(self, order: Order, 
                            market_data: MarketData,
                            constraints: Dict) -> ExecutionPlan:
        """Create execution plan for an order"""
        pass
    
    @abstractmethod
    def get_next_slice(self, plan: ExecutionPlan,
                      market_data: MarketData,
                      current_time: datetime) -> Optional[Order]:
        """Get next order slice to execute"""
        pass
    
    def execute_order(self, order: Order,
                     market_data_stream: MarketDataStream,
                     execution_handler: ExecutionHandler) -> ExecutionResult:
        """Main execution loop"""
        # Create execution plan
        plan = self.create_execution_plan(
            order, 
            market_data_stream.get_snapshot(),
            self._get_default_constraints()
        )
        
        self.active_plans[order.id] = plan
        self.execution_stats['total_orders'] += 1
        
        # Execute according to plan
        while plan.remaining_quantity > 0 and datetime.now() < plan.end_time:
            # Get current market data
            market_data = market_data_stream.get_snapshot()
            
            # Check if we should send next slice
            next_slice = self.get_next_slice(plan, market_data, datetime.now())
            
            if next_slice:
                # Send order
                fill = execution_handler.send_order(next_slice)
                
                # Update plan
                if fill:
                    plan.update_progress(fill)
                    self._update_execution_stats(plan, fill, market_data)
            
            # Wait for next decision point
            time.sleep(self._get_sleep_duration(plan))
        
        # Finalize execution
        return self._finalize_execution(plan)
    
    def _calculate_slippage(self, plan: ExecutionPlan, 
                          market_data: MarketData) -> float:
        """Calculate execution slippage"""
        # Get arrival price (price when order was received)
        arrival_price = plan.parent_order.metadata.get('arrival_price', plan.average_price)
        
        # Calculate slippage in basis points
        if plan.parent_order.side == OrderSide.BUY:
            slippage_bps = (plan.average_price - arrival_price) / arrival_price * 10000
        else:
            slippage_bps = (arrival_price - plan.average_price) / arrival_price * 10000
        
        return slippage_bps
    
    def adapt_to_market_conditions(self, plan: ExecutionPlan,
                                 market_data: MarketData) -> None:
        """Adapt execution strategy based on market conditions"""
        # This method can be overridden by specific algorithms
        pass

class SmartOrderRouter:
    """
    Routes orders to best execution venues.
    Handles venue selection and order splitting.
    """
    
    def __init__(self, venues: List[ExecutionVenue]):
        self.venues = venues
        self.venue_stats = {venue.name: VenueStatistics() for venue in venues}
        self.logger = ComponentLogger("SmartOrderRouter", "execution")
    
    def route_order(self, order: Order, 
                   market_data: Dict[str, MarketData]) -> List[Tuple[ExecutionVenue, Order]]:
        """Route order to optimal venues"""
        
        # Get venue rankings
        venue_scores = self._rank_venues(order, market_data)
        
        # Determine split across venues
        venue_allocations = self._allocate_to_venues(order, venue_scores, market_data)
        
        # Create child orders
        routed_orders = []
        for venue, allocation in venue_allocations.items():
            if allocation['quantity'] > 0:
                child_order = self._create_child_order(
                    order, venue, allocation['quantity'], allocation['price_limit']
                )
                routed_orders.append((venue, child_order))
        
        return routed_orders
    
    def _rank_venues(self, order: Order,
                    market_data: Dict[str, MarketData]) -> Dict[ExecutionVenue, float]:
        """Rank venues by execution quality"""
        scores = {}
        
        for venue in self.venues:
            if venue.name not in market_data:
                continue
            
            venue_data = market_data[venue.name]
            
            # Calculate venue score based on multiple factors
            spread_score = self._calculate_spread_score(venue_data)
            depth_score = self._calculate_depth_score(venue_data, order.quantity)
            fee_score = self._calculate_fee_score(venue, order)
            historical_score = self._calculate_historical_score(venue)
            
            # Weighted combination
            total_score = (
                0.3 * spread_score +
                0.3 * depth_score +
                0.2 * fee_score +
                0.2 * historical_score
            )
            
            scores[venue] = total_score
        
        return scores
    
    def _allocate_to_venues(self, order: Order,
                          venue_scores: Dict[ExecutionVenue, float],
                          market_data: Dict[str, MarketData]) -> Dict[ExecutionVenue, Dict]:
        """Allocate order quantity across venues"""
        allocations = {}
        remaining_quantity = order.quantity
        
        # Sort venues by score
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        
        for venue, score in sorted_venues:
            if remaining_quantity <= 0:
                break
            
            # Calculate allocation for this venue
            venue_data = market_data.get(venue.name)
            if not venue_data:
                continue
            
            # Get available liquidity
            if order.side == OrderSide.BUY:
                available_liquidity = venue_data.ask_size
                price_limit = venue_data.ask_price * 1.001  # 10bps buffer
            else:
                available_liquidity = venue_data.bid_size
                price_limit = venue_data.bid_price * 0.999
            
            # Allocate based on score and liquidity
            max_allocation = min(
                remaining_quantity,
                available_liquidity * 0.2,  # Max 20% of displayed liquidity
                order.quantity * score  # Proportional to score
            )
            
            if max_allocation > venue.min_order_size:
                allocations[venue] = {
                    'quantity': int(max_allocation),
                    'price_limit': price_limit
                }
                remaining_quantity -= max_allocation
        
        return allocations
```

### 2. TWAP Implementation

```python
# src/execution/algorithms/twap.py
class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price algorithm.
    Executes order evenly over time period.
    """
    
    def __init__(self):
        super().__init__("TWAP")
        self.min_slice_interval = timedelta(seconds=30)
        self.randomization_factor = 0.2  # 20% randomization
    
    def create_execution_plan(self, order: Order,
                            market_data: MarketData,
                            constraints: Dict) -> ExecutionPlan:
        """Create TWAP execution plan"""
        
        # Determine execution window
        duration_minutes = constraints.get('duration_minutes', 60)
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Calculate number of slices
        n_slices = max(
            int(duration_minutes * 60 / self.min_slice_interval.total_seconds()),
            10  # Minimum 10 slices
        )
        
        # Create time schedule
        schedule = self._create_time_schedule(
            start_time, end_time, n_slices, order.quantity
        )
        
        # Apply randomization to avoid predictability
        schedule = self._randomize_schedule(schedule, self.randomization_factor)
        
        plan = ExecutionPlan(
            parent_order=order,
            child_orders=[],
            schedule=schedule,
            start_time=start_time,
            end_time=end_time,
            style=ExecutionStyle.NORMAL,
            min_fill_size=constraints.get('min_fill_size', 100),
            max_participation_rate=constraints.get('max_participation', 0.1),
            price_limit=constraints.get('price_limit'),
            remaining_quantity=order.quantity
        )
        
        return plan
    
    def _create_time_schedule(self, start_time: datetime,
                            end_time: datetime,
                            n_slices: int,
                            total_quantity: float) -> pd.DataFrame:
        """Create evenly distributed time schedule"""
        
        # Generate time points
        time_points = pd.date_range(start_time, end_time, periods=n_slices + 1)
        
        # Calculate quantities
        base_quantity = total_quantity / n_slices
        quantities = [base_quantity] * n_slices
        
        # Adjust last slice for rounding
        quantities[-1] = total_quantity - sum(quantities[:-1])
        
        schedule = pd.DataFrame({
            'time': time_points[:-1],
            'quantity': quantities,
            'executed': [False] * n_slices,
            'actual_quantity': [0] * n_slices,
            'price': [0] * n_slices
        })
        
        return schedule
    
    def _randomize_schedule(self, schedule: pd.DataFrame,
                          randomization_factor: float) -> pd.DataFrame:
        """Add randomization to avoid detection"""
        
        # Randomize quantities
        random_factors = np.random.uniform(
            1 - randomization_factor,
            1 + randomization_factor,
            len(schedule)
        )
        
        schedule['quantity'] = schedule['quantity'] * random_factors
        
        # Ensure total quantity is preserved
        total_quantity = schedule['quantity'].sum()
        schedule['quantity'] = schedule['quantity'] / schedule['quantity'].sum() * total_quantity
        
        # Randomize times slightly
        time_jitter = pd.to_timedelta(
            np.random.uniform(-30, 30, len(schedule)), 
            unit='seconds'
        )
        schedule['time'] = schedule['time'] + time_jitter
        
        return schedule.sort_values('time').reset_index(drop=True)
    
    def get_next_slice(self, plan: ExecutionPlan,
                      market_data: MarketData,
                      current_time: datetime) -> Optional[Order]:
        """Get next TWAP slice"""
        
        # Find next scheduled slice
        pending_slices = plan.schedule[~plan.schedule['executed']]
        if pending_slices.empty:
            return None
        
        next_slice = pending_slices.iloc[0]
        
        # Check if it's time to execute
        if current_time >= next_slice['time']:
            # Adjust quantity based on progress
            adjusted_quantity = self._adjust_slice_quantity(
                plan, next_slice['quantity'], market_data
            )
            
            if adjusted_quantity > plan.min_fill_size:
                # Create child order
                child_order = Order(
                    asset=plan.parent_order.asset,
                    quantity=adjusted_quantity,
                    side=plan.parent_order.side,
                    order_type=OrderType.LIMIT,
                    limit_price=self._calculate_limit_price(
                        plan.parent_order.side, market_data
                    ),
                    metadata={
                        'parent_id': plan.parent_order.id,
                        'algorithm': 'TWAP',
                        'slice_index': next_slice.name
                    }
                )
                
                # Mark as executed
                plan.schedule.loc[next_slice.name, 'executed'] = True
                
                return child_order
        
        return None
    
    def _adjust_slice_quantity(self, plan: ExecutionPlan,
                             base_quantity: float,
                             market_data: MarketData) -> float:
        """Adjust slice quantity based on execution progress"""
        
        # Calculate catch-up if behind schedule
        scheduled_progress = len(plan.schedule[plan.schedule['executed']]) / len(plan.schedule)
        actual_progress = plan.executed_quantity / plan.parent_order.quantity
        
        if actual_progress < scheduled_progress - 0.1:  # More than 10% behind
            # Increase slice size to catch up
            adjustment_factor = 1.5
        elif actual_progress > scheduled_progress + 0.1:  # More than 10% ahead
            # Decrease slice size
            adjustment_factor = 0.7
        else:
            adjustment_factor = 1.0
        
        adjusted_quantity = base_quantity * adjustment_factor
        
        # Apply participation rate limit
        market_volume = market_data.volume_1min
        max_quantity = market_volume * plan.max_participation_rate
        
        return min(adjusted_quantity, max_quantity, plan.remaining_quantity)
```

### 3. VWAP Implementation

```python
# src/execution/algorithms/vwap.py
class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price algorithm.
    Executes in proportion to market volume.
    """
    
    def __init__(self):
        super().__init__("VWAP")
        self.volume_predictor = VolumePredictor()
        self.min_participation = 0.01
        self.max_participation = 0.15
    
    def create_execution_plan(self, order: Order,
                            market_data: MarketData,
                            constraints: Dict) -> ExecutionPlan:
        """Create VWAP execution plan"""
        
        # Get historical volume profile
        volume_profile = self.volume_predictor.get_intraday_profile(
            order.asset,
            lookback_days=20
        )
        
        # Determine execution window
        start_time = datetime.now()
        end_time = self._calculate_end_time(start_time, constraints)
        
        # Create volume-based schedule
        schedule = self._create_volume_schedule(
            start_time, end_time, order.quantity, volume_profile
        )
        
        plan = ExecutionPlan(
            parent_order=order,
            child_orders=[],
            schedule=schedule,
            start_time=start_time,
            end_time=end_time,
            style=ExecutionStyle.PASSIVE,
            min_fill_size=constraints.get('min_fill_size', 100),
            max_participation_rate=constraints.get('max_participation', self.max_participation),
            price_limit=constraints.get('price_limit'),
            remaining_quantity=order.quantity
        )
        
        return plan
    
    def _create_volume_schedule(self, start_time: datetime,
                              end_time: datetime,
                              total_quantity: float,
                              volume_profile: pd.DataFrame) -> pd.DataFrame:
        """Create schedule based on volume profile"""
        
        # Get time range
        start_minutes = start_time.hour * 60 + start_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute
        
        # Filter volume profile to execution window
        window_profile = volume_profile[
            (volume_profile['minute'] >= start_minutes) &
            (volume_profile['minute'] <= end_minutes)
        ].copy()
        
        # Normalize to get distribution
        window_profile['volume_pct'] = (
            window_profile['avg_volume'] / window_profile['avg_volume'].sum()
        )
        
        # Allocate quantity based on volume
        window_profile['target_quantity'] = total_quantity * window_profile['volume_pct']
        
        # Convert to schedule
        schedule_data = []
        for _, row in window_profile.iterrows():
            time = start_time.replace(
                hour=int(row['minute'] // 60),
                minute=int(row['minute'] % 60),
                second=0,
                microsecond=0
            )
            
            schedule_data.append({
                'time': time,
                'quantity': row['target_quantity'],
                'expected_volume': row['avg_volume'],
                'volume_pct': row['volume_pct'],
                'executed': False,
                'actual_quantity': 0,
                'actual_volume': 0
            })
        
        return pd.DataFrame(schedule_data)
    
    def get_next_slice(self, plan: ExecutionPlan,
                      market_data: MarketData,
                      current_time: datetime) -> Optional[Order]:
        """Get next VWAP slice based on volume"""
        
        # Update actual volume in schedule
        self._update_volume_tracking(plan, market_data)
        
        # Calculate participation based on real-time volume
        current_minute = current_time.hour * 60 + current_time.minute
        
        # Find current time slot
        current_slots = plan.schedule[
            plan.schedule['time'].dt.hour * 60 + 
            plan.schedule['time'].dt.minute == current_minute
        ]
        
        if current_slots.empty:
            return None
        
        current_slot = current_slots.iloc[0]
        
        # Calculate how much we should have executed by now
        elapsed_seconds = (current_time - current_slot['time']).total_seconds()
        time_progress = min(elapsed_seconds / 60, 1.0)  # Progress through minute
        
        # Target quantity based on volume
        actual_volume = market_data.volume_1min
        participation_rate = min(
            current_slot['quantity'] / max(actual_volume, 1),
            plan.max_participation_rate
        )
        
        target_quantity = actual_volume * participation_rate * time_progress
        executed_quantity = current_slot['actual_quantity']
        
        slice_quantity = target_quantity - executed_quantity
        
        if slice_quantity > plan.min_fill_size:
            # Create order
            child_order = Order(
                asset=plan.parent_order.asset,
                quantity=int(slice_quantity),
                side=plan.parent_order.side,
                order_type=OrderType.LIMIT,
                limit_price=self._calculate_adaptive_limit_price(
                    plan.parent_order.side, market_data, participation_rate
                ),
                metadata={
                    'parent_id': plan.parent_order.id,
                    'algorithm': 'VWAP',
                    'participation_rate': participation_rate,
                    'minute': current_minute
                }
            )
            
            return child_order
        
        return None
    
    def _calculate_adaptive_limit_price(self, side: OrderSide,
                                      market_data: MarketData,
                                      participation_rate: float) -> float:
        """Calculate limit price based on participation rate"""
        
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        
        # More aggressive pricing for higher participation
        aggressiveness = min(participation_rate / self.max_participation, 1.0)
        
        if side == OrderSide.BUY:
            # Start at mid, move toward ask as participation increases
            limit_price = mid_price + spread * 0.5 * aggressiveness
        else:
            # Start at mid, move toward bid as participation increases
            limit_price = mid_price - spread * 0.5 * aggressiveness
        
        return limit_price

class VolumePredictor:
    """Predicts intraday volume patterns"""
    
    def __init__(self):
        self.volume_cache = {}
        self.logger = ComponentLogger("VolumePredictor", "execution")
    
    def get_intraday_profile(self, asset: str, 
                           lookback_days: int = 20) -> pd.DataFrame:
        """Get average intraday volume profile"""
        
        cache_key = f"{asset}_{lookback_days}"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        
        # Load historical data
        historical_data = self._load_historical_minutes(asset, lookback_days)
        
        # Calculate average volume by minute of day
        historical_data['minute'] = (
            historical_data.index.hour * 60 + 
            historical_data.index.minute
        )
        
        volume_profile = historical_data.groupby('minute').agg({
            'volume': ['mean', 'std', 'median']
        }).reset_index()
        
        volume_profile.columns = ['minute', 'avg_volume', 'std_volume', 'median_volume']
        
        # Smooth the profile
        volume_profile['avg_volume'] = (
            volume_profile['avg_volume'].rolling(window=5, center=True).mean()
            .fillna(method='bfill').fillna(method='ffill')
        )
        
        self.volume_cache[cache_key] = volume_profile
        
        return volume_profile
```

### 4. Implementation Shortfall Minimization

```python
# src/execution/algorithms/implementation_shortfall.py
class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """
    Minimizes implementation shortfall (slippage + opportunity cost).
    Balances urgency against market impact.
    """
    
    def __init__(self):
        super().__init__("IS")
        self.impact_model = MarketImpactModel()
        self.alpha_decay_model = AlphaDecayModel()
        self.optimizer = ISOptimizer()
    
    def create_execution_plan(self, order: Order,
                            market_data: MarketData,
                            constraints: Dict) -> ExecutionPlan:
        """Create optimal execution plan to minimize IS"""
        
        # Estimate market impact parameters
        impact_params = self.impact_model.estimate_parameters(
            order.asset,
            market_data,
            lookback_days=30
        )
        
        # Estimate alpha decay
        alpha_params = self.alpha_decay_model.estimate_decay(
            order.asset,
            order.metadata.get('signal_strength', 1.0)
        )
        
        # Optimize execution trajectory
        optimal_trajectory = self.optimizer.optimize_trajectory(
            order_size=order.quantity,
            market_params=impact_params,
            alpha_params=alpha_params,
            risk_aversion=constraints.get('risk_aversion', 1.0)
        )
        
        # Convert to schedule
        schedule = self._trajectory_to_schedule(
            optimal_trajectory,
            order.quantity,
            datetime.now()
        )
        
        plan = ExecutionPlan(
            parent_order=order,
            child_orders=[],
            schedule=schedule,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=optimal_trajectory['duration']),
            style=ExecutionStyle.ADAPTIVE,
            min_fill_size=constraints.get('min_fill_size', 100),
            max_participation_rate=0.2,  # Higher for IS
            price_limit=constraints.get('price_limit'),
            remaining_quantity=order.quantity
        )
        
        return plan
    
    def get_next_slice(self, plan: ExecutionPlan,
                      market_data: MarketData,
                      current_time: datetime) -> Optional[Order]:
        """Get next slice with dynamic adjustment"""
        
        # Update market conditions
        current_volatility = self._estimate_current_volatility(market_data)
        current_liquidity = self._estimate_current_liquidity(market_data)
        
        # Adjust trajectory if needed
        if self._should_reoptimize(plan, current_volatility, current_liquidity):
            self._reoptimize_trajectory(plan, market_data)
        
        # Get scheduled quantity
        elapsed_time = (current_time - plan.start_time).total_seconds() / 60
        target_progress = self._get_target_progress(plan.schedule, elapsed_time)
        current_progress = plan.executed_quantity / plan.parent_order.quantity
        
        # Calculate slice size
        progress_gap = target_progress - current_progress
        urgency_factor = self._calculate_urgency(progress_gap, elapsed_time, plan)
        
        base_quantity = plan.remaining_quantity * urgency_factor
        adjusted_quantity = self._adjust_for_market_conditions(
            base_quantity, market_data, plan
        )
        
        if adjusted_quantity > plan.min_fill_size:
            # Determine order type and pricing
            order_type, limit_price = self._determine_order_params(
                adjusted_quantity, market_data, urgency_factor
            )
            
            child_order = Order(
                asset=plan.parent_order.asset,
                quantity=int(adjusted_quantity),
                side=plan.parent_order.side,
                order_type=order_type,
                limit_price=limit_price,
                metadata={
                    'parent_id': plan.parent_order.id,
                    'algorithm': 'IS',
                    'urgency': urgency_factor,
                    'expected_impact': self._estimate_impact(adjusted_quantity, market_data)
                }
            )
            
            return child_order
        
        return None
    
    def _calculate_urgency(self, progress_gap: float, 
                         elapsed_time: float,
                         plan: ExecutionPlan) -> float:
        """Calculate urgency factor based on progress and alpha decay"""
        
        # Base urgency from progress gap
        if progress_gap > 0.1:  # Behind schedule
            progress_urgency = min(2.0, 1 + progress_gap * 2)
        else:
            progress_urgency = max(0.5, 1 + progress_gap)
        
        # Alpha decay urgency
        remaining_alpha = self.alpha_decay_model.get_remaining_alpha(
            elapsed_time,
            plan.parent_order.metadata.get('alpha_half_life', 60)
        )
        
        # Higher urgency if alpha is decaying quickly
        if remaining_alpha < 0.5:
            alpha_urgency = 1.5
        elif remaining_alpha < 0.3:
            alpha_urgency = 2.0
        else:
            alpha_urgency = 1.0
        
        return progress_urgency * alpha_urgency

class MarketImpactModel:
    """Models temporary and permanent market impact"""
    
    def estimate_parameters(self, asset: str,
                          market_data: MarketData,
                          lookback_days: int = 30) -> Dict:
        """Estimate market impact parameters"""
        
        # Load historical data
        historical_data = self._load_historical_data(asset, lookback_days)
        
        # Estimate temporary impact (Kyle's lambda)
        temp_impact = self._estimate_kyle_lambda(historical_data)
        
        # Estimate permanent impact
        perm_impact = self._estimate_permanent_impact(historical_data)
        
        # Estimate decay rate
        decay_rate = self._estimate_impact_decay(historical_data)
        
        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'decay_rate': decay_rate,
            'volatility': historical_data['returns'].std() * np.sqrt(252),
            'avg_spread': historical_data['spread'].mean(),
            'avg_volume': historical_data['volume'].mean()
        }
    
class ISOptimizer:
    """Optimizes execution trajectory for implementation shortfall"""
    
    def optimize_trajectory(self, order_size: float,
                          market_params: Dict,
                          alpha_params: Dict,
                          risk_aversion: float) -> Dict:
        """Solve for optimal execution trajectory"""
        
        # Almgren-Chriss framework
        volatility = market_params['volatility']
        temp_impact = market_params['temporary_impact']
        perm_impact = market_params['permanent_impact']
        
        # Risk aversion parameter
        lambda_risk = risk_aversion
        
        # Optimal execution time (simplified)
        T_opt = np.sqrt(
            3 * temp_impact * order_size / 
            (2 * lambda_risk * volatility**2)
        )
        
        # Cap at reasonable duration
        T_opt = min(T_opt, 240)  # Max 4 hours
        
        # Generate trajectory
        n_steps = max(int(T_opt / 5), 10)  # 5-minute intervals
        times = np.linspace(0, T_opt, n_steps)
        
        # Optimal trading rate (simplified linear for now)
        # Real implementation would use Almgren-Chriss solution
        trading_rates = np.ones(n_steps) / n_steps
        
        return {
            'duration': T_opt,
            'times': times,
            'trading_rates': trading_rates,
            'expected_cost': self._calculate_expected_cost(
                order_size, T_opt, market_params
            )
        }
```

### 5. Smart Order Types

```python
# src/execution/algorithms/smart_orders.py
class IcebergOrder(ExecutionAlgorithm):
    """
    Iceberg order - shows only small visible quantity.
    Hides true order size.
    """
    
    def __init__(self, visible_ratio: float = 0.1):
        super().__init__("Iceberg")
        self.visible_ratio = visible_ratio
        self.min_visible_size = 100
        self.refresh_ratio = 0.8  # Refresh when 80% filled
    
    def execute_iceberg(self, order: Order,
                       execution_handler: ExecutionHandler) -> ExecutionResult:
        """Execute iceberg order"""
        
        total_quantity = order.quantity
        visible_quantity = max(
            int(total_quantity * self.visible_ratio),
            self.min_visible_size
        )
        
        executed_quantity = 0
        fills = []
        
        while executed_quantity < total_quantity:
            # Calculate next visible slice
            remaining = total_quantity - executed_quantity
            current_visible = min(visible_quantity, remaining)
            
            # Create visible order
            visible_order = Order(
                asset=order.asset,
                quantity=current_visible,
                side=order.side,
                order_type=OrderType.LIMIT,
                limit_price=order.limit_price,
                metadata={
                    'parent_id': order.id,
                    'order_type': 'iceberg',
                    'total_quantity': total_quantity,
                    'visible_quantity': current_visible
                }
            )
            
            # Send and monitor
            result = execution_handler.send_order(visible_order)
            
            if result.status == ExecutionStatus.FILLED:
                executed_quantity += result.filled_quantity
                fills.append(result)
                
                # Adjust price if needed
                if self._should_adjust_price(fills, order):
                    order.limit_price = self._calculate_new_price(
                        order, execution_handler.get_market_data()
                    )
            
            elif result.status == ExecutionStatus.CANCELLED:
                break
            
            # Small delay to avoid detection
            time.sleep(random.uniform(0.5, 2.0))
        
        return self._aggregate_fills(fills)

class AdaptiveOrder:
    """
    Adaptive order that changes behavior based on market conditions.
    Switches between passive and aggressive modes.
    """
    
    def __init__(self):
        self.passive_algo = TWAPAlgorithm()
        self.aggressive_algo = ImplementationShortfallAlgorithm()
        self.market_analyzer = MarketConditionAnalyzer()
        self.logger = ComponentLogger("AdaptiveOrder", "execution")
    
    def execute(self, order: Order,
               market_data_stream: MarketDataStream,
               execution_handler: ExecutionHandler) -> ExecutionResult:
        """Execute with adaptive behavior"""
        
        # Initial market assessment
        market_conditions = self.market_analyzer.assess_conditions(
            market_data_stream.get_snapshot()
        )
        
        # Choose initial algorithm
        if market_conditions['volatility'] > 0.02 or market_conditions['liquidity'] < 0.5:
            current_algo = self.aggressive_algo
            self.logger.info("Starting with aggressive execution due to market conditions")
        else:
            current_algo = self.passive_algo
            self.logger.info("Starting with passive execution")
        
        # Create execution plan
        plan = current_algo.create_execution_plan(
            order,
            market_data_stream.get_snapshot(),
            {}
        )
        
        # Execute with periodic reassessment
        while plan.remaining_quantity > 0:
            # Check if we should switch algorithms
            new_conditions = self.market_analyzer.assess_conditions(
                market_data_stream.get_snapshot()
            )
            
            if self._should_switch_algorithm(market_conditions, new_conditions, plan):
                # Switch algorithm
                current_algo = self._select_algorithm(new_conditions, plan)
                
                # Recreate plan with new algorithm
                remaining_order = Order(
                    asset=order.asset,
                    quantity=plan.remaining_quantity,
                    side=order.side,
                    order_type=order.order_type,
                    metadata=order.metadata
                )
                
                plan = current_algo.create_execution_plan(
                    remaining_order,
                    market_data_stream.get_snapshot(),
                    {}
                )
            
            # Execute next slice
            next_slice = current_algo.get_next_slice(
                plan,
                market_data_stream.get_snapshot(),
                datetime.now()
            )
            
            if next_slice:
                fill = execution_handler.send_order(next_slice)
                if fill:
                    plan.update_progress(fill)
            
            market_conditions = new_conditions
            time.sleep(1)
        
        return self._create_execution_result(plan)
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_3_execution_algos.py`:

```python
class TestExecutionAlgorithms:
    """Test execution algorithm implementations"""
    
    def test_twap_schedule_creation(self):
        """Test TWAP schedule generation"""
        algo = TWAPAlgorithm()
        
        order = Order(
            asset='SPY',
            quantity=10000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        
        market_data = create_test_market_data()
        
        plan = algo.create_execution_plan(
            order,
            market_data,
            {'duration_minutes': 60}
        )
        
        # Verify schedule
        assert len(plan.schedule) >= 10  # At least 10 slices
        assert abs(plan.schedule['quantity'].sum() - 10000) < 1  # Total quantity preserved
        assert plan.schedule['time'].is_monotonic_increasing  # Times in order
    
    def test_vwap_volume_profile(self):
        """Test VWAP volume profile calculation"""
        predictor = VolumePredictor()
        
        # Create synthetic intraday volume pattern
        minutes = range(390)  # 6.5 hours of trading
        volumes = [
            1000000 * (1 + 0.5 * np.sin(m / 390 * 2 * np.pi))  # U-shaped
            for m in minutes
        ]
        
        profile = predictor._create_volume_profile(minutes, volumes)
        
        # Verify U-shape (higher at open/close)
        assert profile.iloc[0]['avg_volume'] > profile.iloc[195]['avg_volume']
        assert profile.iloc[-1]['avg_volume'] > profile.iloc[195]['avg_volume']
    
    def test_implementation_shortfall_urgency(self):
        """Test IS urgency calculation"""
        algo = ImplementationShortfallAlgorithm()
        
        plan = create_test_execution_plan()
        
        # Test urgency increases when behind schedule
        urgency_behind = algo._calculate_urgency(
            progress_gap=0.2,  # 20% behind
            elapsed_time=30,
            plan=plan
        )
        
        urgency_on_track = algo._calculate_urgency(
            progress_gap=0.0,
            elapsed_time=30,
            plan=plan
        )
        
        assert urgency_behind > urgency_on_track

class TestSmartOrderRouter:
    """Test smart order routing"""
    
    def test_venue_ranking(self):
        """Test venue scoring and ranking"""
        venues = [
            ExecutionVenue('NYSE', min_order_size=100, fee_rate=0.0025),
            ExecutionVenue('NASDAQ', min_order_size=1, fee_rate=0.0020),
            ExecutionVenue('DARK', min_order_size=1000, fee_rate=0.0010)
        ]
        
        router = SmartOrderRouter(venues)
        
        order = Order('SPY', 5000, OrderSide.BUY, OrderType.MARKET)
        
        market_data = {
            'NYSE': MarketData(bid=400, ask=400.05, bid_size=1000, ask_size=1500),
            'NASDAQ': MarketData(bid=400.01, ask=400.04, bid_size=2000, ask_size=2500),
            'DARK': MarketData(bid=400.02, ask=400.03, bid_size=5000, ask_size=5000)
        }
        
        scores = router._rank_venues(order, market_data)
        
        # Dark pool should score highest (best spread, lowest fee)
        assert max(scores, key=scores.get).name == 'DARK'
```

### Integration Tests

Create `tests/integration/test_step10_3_execution_integration.py`:

```python
def test_complete_execution_workflow():
    """Test full execution algorithm workflow"""
    # Setup
    order = Order(
        asset='SPY',
        quantity=50000,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        metadata={'arrival_price': 400.00}
    )
    
    # Market data stream
    market_stream = SimulatedMarketDataStream('SPY')
    
    # Execution handler
    exec_handler = SimulatedExecutionHandler(
        fill_ratio=0.8,  # 80% fill rate
        slippage_bps=2   # 2 bps slippage
    )
    
    # Run TWAP execution
    twap = TWAPAlgorithm()
    result = twap.execute_order(order, market_stream, exec_handler)
    
    # Verify execution
    assert result.filled_quantity > 0
    assert result.filled_quantity <= order.quantity
    assert result.average_price > 0
    assert result.slippage_bps < 10  # Reasonable slippage

def test_adaptive_execution():
    """Test adaptive execution under changing conditions"""
    order = Order('SPY', 100000, OrderSide.SELL, OrderType.MARKET)
    
    # Create market stream with changing conditions
    market_stream = create_volatile_market_stream()
    exec_handler = SimulatedExecutionHandler()
    
    # Adaptive execution
    adaptive = AdaptiveOrder()
    result = adaptive.execute(order, market_stream, exec_handler)
    
    # Should complete despite volatility
    assert result.filled_quantity == order.quantity
    
    # Check that algorithm switched (from metadata)
    algo_switches = [
        fill.metadata.get('algorithm') 
        for fill in result.fills
    ]
    assert len(set(algo_switches)) > 1  # Used multiple algorithms

def test_iceberg_order_execution():
    """Test iceberg order hiding"""
    order = Order(
        asset='SPY',
        quantity=100000,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=400.10
    )
    
    iceberg = IcebergOrder(visible_ratio=0.05)  # Show only 5%
    exec_handler = SimulatedExecutionHandler()
    
    # Track visible quantities
    visible_quantities = []
    
    def track_visible(child_order):
        visible_quantities.append(child_order.quantity)
        return exec_handler.send_order(child_order)
    
    exec_handler.send_order = track_visible
    
    result = iceberg.execute_iceberg(order, exec_handler)
    
    # Verify iceberg behavior
    assert all(q <= 5000 for q in visible_quantities)  # Max 5% visible
    assert sum(visible_quantities) >= order.quantity  # All executed
```

### System Tests

Create `tests/system/test_step10_3_production_execution.py`:

```python
def test_high_frequency_execution():
    """Test execution algorithms under high-frequency conditions"""
    # Multiple simultaneous orders
    orders = [
        Order(f'STOCK_{i}', random.randint(1000, 10000), 
              random.choice([OrderSide.BUY, OrderSide.SELL]), OrderType.MARKET)
        for i in range(20)
    ]
    
    # Execution engine with multiple algorithms
    exec_engine = ExecutionEngine()
    exec_engine.register_algorithm('TWAP', TWAPAlgorithm())
    exec_engine.register_algorithm('VWAP', VWAPAlgorithm())
    exec_engine.register_algorithm('IS', ImplementationShortfallAlgorithm())
    
    # Execute all orders concurrently
    results = exec_engine.execute_orders_parallel(
        orders,
        algorithm_selector=lambda o: 'VWAP' if o.quantity > 5000 else 'TWAP'
    )
    
    # Performance checks
    assert len(results) == len(orders)
    assert all(r.status == ExecutionStatus.COMPLETED for r in results)
    
    # Slippage analysis
    slippages = [r.slippage_bps for r in results]
    assert np.mean(slippages) < 5  # Average slippage under 5 bps
    assert np.percentile(slippages, 95) < 10  # 95th percentile under 10 bps

def test_market_impact_estimation():
    """Test market impact model accuracy"""
    impact_model = MarketImpactModel()
    
    # Test with known scenarios
    test_cases = [
        {
            'asset': 'SPY',
            'order_size': 10000,
            'avg_volume': 1000000,
            'volatility': 0.15,
            'expected_impact_bps': 2  # ~2 bps for 1% of volume
        },
        {
            'asset': 'ILLIQUID',
            'order_size': 5000,
            'avg_volume': 50000,
            'volatility': 0.30,
            'expected_impact_bps': 50  # ~50 bps for 10% of volume
        }
    ]
    
    for case in test_cases:
        market_data = create_market_data_for_impact_test(case)
        
        params = impact_model.estimate_parameters(
            case['asset'],
            market_data,
            lookback_days=30
        )
        
        # Estimate impact
        estimated_impact = impact_model.estimate_total_impact(
            case['order_size'],
            params
        )
        
        # Should be within reasonable range
        assert abs(estimated_impact - case['expected_impact_bps']) < case['expected_impact_bps'] * 0.5
```

## ✅ Validation Checklist

### Core Framework
- [ ] Execution plans created correctly
- [ ] Child order generation working
- [ ] Progress tracking accurate
- [ ] Slippage calculation correct

### TWAP Algorithm
- [ ] Even time distribution
- [ ] Randomization applied
- [ ] Catch-up logic working
- [ ] Participation limits enforced

### VWAP Algorithm
- [ ] Volume profile accurate
- [ ] Real-time adaptation working
- [ ] Participation rate controlled
- [ ] Price adjustment logical

### Implementation Shortfall
- [ ] Impact model calibrated
- [ ] Urgency calculation correct
- [ ] Trajectory optimization working
- [ ] Dynamic adaptation functional

### Smart Orders
- [ ] Iceberg hiding effective
- [ ] Adaptive switching working
- [ ] Venue routing optimal
- [ ] Order types correct

## 📊 Performance Benchmarks

### Execution Performance
- Slice generation: < 5ms
- Market data update: < 10ms
- Impact calculation: < 20ms
- Route determination: < 10ms

### Algorithm Accuracy
- TWAP tracking error: < 2%
- VWAP tracking error: < 3%
- IS cost vs estimate: < 20%
- Fill rate: > 95%

### Slippage Targets
- Liquid assets: < 5 bps
- Mid-liquidity: < 10 bps
- Illiquid assets: < 25 bps
- Urgent orders: < 50 bps

## 🐛 Common Issues

1. **Market Impact Underestimation**
   - Calibrate models regularly
   - Account for intraday patterns
   - Consider market regime

2. **Gaming by Others**
   - Randomize execution
   - Vary slice sizes
   - Use multiple venues

3. **Technology Latency**
   - Minimize computation in critical path
   - Pre-calculate when possible
   - Use efficient data structures

## 🎯 Success Criteria

Step 10.3 is complete when:
1. ✅ All execution algorithms implemented
2. ✅ Smart order routing functional
3. ✅ Market impact models calibrated
4. ✅ Performance benchmarks met
5. ✅ Production-ready testing complete

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.4: Market Making](step-10.4-market-making.md)

## 📚 Additional Resources

- [Algorithmic Trading and DMA](../references/algo-trading-dma.md)
- [Optimal Execution Theory](../references/optimal-execution.md)
- [Market Microstructure Practitioner Guide](../references/microstructure-practice.md)
- [Transaction Cost Analysis](../references/tca-analysis.md)