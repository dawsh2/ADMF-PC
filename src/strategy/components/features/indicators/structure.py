"""
Structure-based incremental features.

Features that detect price structure patterns, levels, and geometric relationships.
All features maintain state and update in O(1) time complexity.

Uses protocol + composition architecture - no inheritance.
"""

import math
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from ..protocols import Feature, FeatureState


class PivotPoints:
    """Pivot Points with O(1) updates. Classic pivot points calculation."""
    
    def __init__(self, name: str = "pivot"):
        self._state = FeatureState(name)
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("Pivot Points require high and low prices")
        
        if self._prev_high is not None and self._prev_low is not None and self._prev_close is not None:
            pp = (self._prev_high + self._prev_low + self._prev_close) / 3
            r1 = 2 * pp - self._prev_low
            s1 = 2 * pp - self._prev_high
            r2 = pp + (self._prev_high - self._prev_low)
            s2 = pp - (self._prev_high - self._prev_low)
            r3 = self._prev_high + 2 * (pp - self._prev_low)
            s3 = self._prev_low - 2 * (self._prev_high - pp)
            
            self._state.set_value({
                "pivot": pp, "r1": r1, "r2": r2, "r3": r3,
                "s1": s1, "s2": s2, "s3": s3
            })
        
        self._prev_high = high
        self._prev_low = low
        self._prev_close = price
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None


class SupportResistance:
    """Support/Resistance levels with O(1) updates."""
    
    def __init__(self, lookback: int = 50, min_touches: int = 2, name: str = "sr"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self.min_touches = min_touches
        self._high_buffer = deque(maxlen=lookback)
        self._low_buffer = deque(maxlen=lookback)
        self._pivot_highs: List[float] = []
        self._pivot_lows: List[float] = []
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("Support/Resistance requires high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) >= 5:
            # Check for pivot high (higher than 2 bars on each side)
            if (len(self._high_buffer) >= 5 and 
                self._high_buffer[-3] > self._high_buffer[-5] and
                self._high_buffer[-3] > self._high_buffer[-4] and
                self._high_buffer[-3] > self._high_buffer[-2] and
                self._high_buffer[-3] > self._high_buffer[-1]):
                self._pivot_highs.append(self._high_buffer[-3])
                if len(self._pivot_highs) > 10:
                    self._pivot_highs.pop(0)
            
            # Check for pivot low
            if (len(self._low_buffer) >= 5 and
                self._low_buffer[-3] < self._low_buffer[-5] and
                self._low_buffer[-3] < self._low_buffer[-4] and
                self._low_buffer[-3] < self._low_buffer[-2] and
                self._low_buffer[-3] < self._low_buffer[-1]):
                self._pivot_lows.append(self._low_buffer[-3])
                if len(self._pivot_lows) > 10:
                    self._pivot_lows.pop(0)
            
            # Find nearest support and resistance
            resistance = None
            support = None
            
            if self._pivot_highs:
                above = [p for p in self._pivot_highs if p > price]
                if above:
                    resistance = min(above)
            
            if self._pivot_lows:
                below = [p for p in self._pivot_lows if p < price]
                if below:
                    support = max(below)
            
            if resistance is not None or support is not None:
                self._state.set_value({
                    "resistance": resistance,
                    "support": support
                })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._pivot_highs.clear()
        self._pivot_lows.clear()


class SwingPoints:
    """Swing High/Low points with O(1) updates."""
    
    def __init__(self, lookback: int = 5, name: str = "swing"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self._high_buffer = deque(maxlen=lookback * 2 + 1)
        self._low_buffer = deque(maxlen=lookback * 2 + 1)
        self._last_swing_high: Optional[float] = None
        self._last_swing_low: Optional[float] = None
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("Swing Points require high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) == self.lookback * 2 + 1:
            mid_idx = self.lookback
            mid_high = self._high_buffer[mid_idx]
            mid_low = self._low_buffer[mid_idx]
            
            # Check for swing high
            is_swing_high = all(mid_high >= self._high_buffer[i] for i in range(len(self._high_buffer)) if i != mid_idx)
            if is_swing_high:
                self._last_swing_high = mid_high
            
            # Check for swing low
            is_swing_low = all(mid_low <= self._low_buffer[i] for i in range(len(self._low_buffer)) if i != mid_idx)
            if is_swing_low:
                self._last_swing_low = mid_low
            
            if self._last_swing_high is not None or self._last_swing_low is not None:
                self._state.set_value({
                    "swing_high": self._last_swing_high,
                    "swing_low": self._last_swing_low
                })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._last_swing_high = None
        self._last_swing_low = None


class LinearRegression:
    """Linear Regression with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "linreg"):
        self._state = FeatureState(name)
        self.period = period
        self._price_buffer = deque(maxlen=period)
        self._x_values = list(range(period))
        self._x_sum = sum(self._x_values)
        self._x_squared_sum = sum(x**2 for x in self._x_values)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, **kwargs) -> Optional[Dict[str, float]]:
        self._price_buffer.append(price)
        
        if len(self._price_buffer) == self.period:
            y_sum = sum(self._price_buffer)
            xy_sum = sum(x * y for x, y in zip(self._x_values, self._price_buffer))
            
            n = self.period
            denominator = n * self._x_squared_sum - self._x_sum ** 2
            
            if denominator != 0:
                slope = (n * xy_sum - self._x_sum * y_sum) / denominator
                intercept = (y_sum - slope * self._x_sum) / n
                current_value = slope * (self.period - 1) + intercept
                
                # Calculate R-squared
                y_mean = y_sum / n
                ss_tot = sum((y - y_mean) ** 2 for y in self._price_buffer)
                ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(self._x_values, self._price_buffer))
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Return multi-value result with slope, intercept, and r_squared
                result = {
                    'value': current_value,
                    'slope': slope,
                    'intercept': intercept,
                    'r2': r_squared
                }
                self._state.set_value(result)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._price_buffer.clear()


class FibonacciRetracement:
    """Fibonacci Retracement levels with O(1) updates."""
    
    def __init__(self, lookback: int = 50, name: str = "fib"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self._high_buffer = deque(maxlen=lookback)
        self._low_buffer = deque(maxlen=lookback)
        self._fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("Fibonacci requires high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) == self.lookback:
            highest = max(self._high_buffer)
            lowest = min(self._low_buffer)
            diff = highest - lowest
            
            if diff > 0:
                levels = {}
                for level in self._fib_levels:
                    levels[f"fib_{int(level*100)}"] = lowest + (diff * level)
                self._state.set_value(levels)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()


class TrendLines:
    """Dynamic Trendline detection with validation and break detection."""
    
    def __init__(self, pivot_lookback: int = 20, min_touches: int = 2, 
                 tolerance: float = 0.002, max_lines: int = 10, num_pivots: int = 3, name: str = "trendlines"):
        self._state = FeatureState(name)
        self.pivot_lookback = pivot_lookback
        self.min_touches = min_touches
        self.tolerance = tolerance
        self.max_lines = max_lines
        self.num_pivots = num_pivots  # Number of pivot points to store
        
        self._high_buffer = deque(maxlen=pivot_lookback * 4)  # Larger buffer for pivot detection
        self._low_buffer = deque(maxlen=pivot_lookback * 4)
        self._close_buffer = deque(maxlen=pivot_lookback * 4)
        self._bar_index = 0
        self._pivot_highs = deque(maxlen=num_pivots)  # Store recent pivot highs
        self._pivot_lows = deque(maxlen=num_pivots)   # Store recent pivot lows
        self._valid_uptrend_lines: List[Dict[str, Any]] = []
        self._valid_downtrend_lines: List[Dict[str, Any]] = []
        
        # Bounce tracking
        self._prev_price = None
        self._prev_low = None
        self._prev_high = None
        self._support_bounces = 0
        self._resistance_bounces = 0
        self._last_support_touch_bar = -1
        self._last_resistance_touch_bar = -1
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if high is None or low is None:
            raise ValueError("TrendLines require high and low prices")
        
        self._bar_index += 1
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        self._close_buffer.append(price)
        
        # Pivot detection similar to Pine Script
        lookback = min(self.pivot_lookback, 3)  # Use practical lookback
        
        if len(self._high_buffer) >= 2 * lookback + 1:
            buffer_len = len(self._high_buffer)
            mid_idx = buffer_len - lookback - 1
            
            if mid_idx >= lookback:
                mid_high = self._high_buffer[mid_idx]
                mid_low = self._low_buffer[mid_idx]
                
                # Check for pivot high
                is_pivot_high = all(
                    self._high_buffer[mid_idx - i] < mid_high and 
                    self._high_buffer[mid_idx + i] < mid_high
                    for i in range(1, lookback + 1)
                )
                
                if is_pivot_high:
                    pivot_bar_idx = self._bar_index - (buffer_len - 1 - mid_idx)
                    self._pivot_highs.append((pivot_bar_idx, mid_high))
                
                # Check for pivot low
                is_pivot_low = all(
                    self._low_buffer[mid_idx - i] > mid_low and 
                    self._low_buffer[mid_idx + i] > mid_low
                    for i in range(1, lookback + 1)
                )
                
                if is_pivot_low:
                    pivot_bar_idx = self._bar_index - (buffer_len - 1 - mid_idx)
                    self._pivot_lows.append((pivot_bar_idx, mid_low))
        
        # Calculate trendlines and find nearest levels
        nearest_support, nearest_resistance = self._calculate_trendlines(price)
        
        # Count valid trend lines
        valid_uptrends = len(self._valid_uptrend_lines)
        valid_downtrends = len(self._valid_downtrend_lines)
        
        # Detect bounces
        self._detect_bounces(price, high, low, nearest_support, nearest_resistance)
        
        # Get slope information
        support_slope, resistance_slope = self._get_nearest_slopes(nearest_support, nearest_resistance)
        
        # Detect branching patterns
        uptrend_branches, downtrend_branches = self._detect_branches()
        
        if nearest_support is not None or nearest_resistance is not None:
            self._state.set_value({
                "valid_uptrends": valid_uptrends,
                "valid_downtrends": valid_downtrends,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_slope": support_slope,
                "resistance_slope": resistance_slope,
                "support_angle": math.degrees(math.atan(support_slope)) if support_slope is not None else None,
                "resistance_angle": math.degrees(math.atan(resistance_slope)) if resistance_slope is not None else None,
                "strongest_uptrend": 1.0 if valid_uptrends > 0 else 0.0,
                "strongest_downtrend": 1.0 if valid_downtrends > 0 else 0.0,
                "support_bounces": self._support_bounces,
                "resistance_bounces": self._resistance_bounces,
                "uptrend_branches": uptrend_branches,
                "downtrend_branches": downtrend_branches
            })
        
        # Update previous values for next iteration
        self._prev_price = price
        self._prev_low = low
        self._prev_high = high
        
        return self._state.value
    
    def _calculate_trendlines(self, current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate valid trendlines and return nearest support/resistance."""
        self._valid_uptrend_lines = []
        self._valid_downtrend_lines = []
        
        # Check uptrend lines (connecting pivot lows)
        if len(self._pivot_lows) >= 2:
            for i in range(len(self._pivot_lows) - 1):
                for j in range(i + 1, len(self._pivot_lows)):
                    idx1, val1 = self._pivot_lows[i]
                    idx2, val2 = self._pivot_lows[j]
                    
                    if idx1 != idx2 and val1 > val2:  # Uptrend: newer pivot is higher
                        slope = (val1 - val2) / (idx1 - idx2)
                        
                        # Validate the trendline
                        valid = True
                        touches = 0
                        
                        # Check if price stayed above the trendline
                        for k in range(max(0, len(self._close_buffer) - (self._bar_index - idx2))):
                            expected_val = val2 + slope * (self._bar_index - k - idx2)
                            if self._close_buffer[-(k+1)] < expected_val * (1 - self.tolerance):
                                valid = False
                                break
                            if abs(self._close_buffer[-(k+1)] - expected_val) / expected_val < self.tolerance:
                                touches += 1
                        
                        if valid and touches >= self.min_touches:
                            current_trendline_val = val2 + slope * (self._bar_index - idx2)
                            self._valid_uptrend_lines.append({
                                'start_idx': idx2,
                                'start_val': val2,
                                'slope': slope,
                                'current_val': current_trendline_val,
                                'touches': touches
                            })
        
        # Check downtrend lines (connecting pivot highs)
        if len(self._pivot_highs) >= 2:
            for i in range(len(self._pivot_highs) - 1):
                for j in range(i + 1, len(self._pivot_highs)):
                    idx1, val1 = self._pivot_highs[i]
                    idx2, val2 = self._pivot_highs[j]
                    
                    if idx1 != idx2 and val1 < val2:  # Downtrend: newer pivot is lower
                        slope = (val2 - val1) / (idx1 - idx2)
                        
                        # Validate the trendline
                        valid = True
                        touches = 0
                        
                        # Check if price stayed below the trendline
                        for k in range(max(0, len(self._close_buffer) - (self._bar_index - idx2))):
                            expected_val = val2 - slope * (self._bar_index - k - idx2)
                            if self._close_buffer[-(k+1)] > expected_val * (1 + self.tolerance):
                                valid = False
                                break
                            if abs(self._close_buffer[-(k+1)] - expected_val) / expected_val < self.tolerance:
                                touches += 1
                        
                        if valid and touches >= self.min_touches:
                            current_trendline_val = val2 - slope * (self._bar_index - idx2)
                            self._valid_downtrend_lines.append({
                                'start_idx': idx2,
                                'start_val': val2,
                                'slope': slope,
                                'current_val': current_trendline_val,
                                'touches': touches
                            })
        
        # Find nearest support and resistance
        nearest_support = None
        nearest_resistance = None
        
        # Support from uptrend lines
        for line in self._valid_uptrend_lines:
            if line['current_val'] < current_price:
                if nearest_support is None or line['current_val'] > nearest_support:
                    nearest_support = line['current_val']
        
        # Resistance from downtrend lines
        for line in self._valid_downtrend_lines:
            if line['current_val'] > current_price:
                if nearest_resistance is None or line['current_val'] < nearest_resistance:
                    nearest_resistance = line['current_val']
        
        return nearest_support, nearest_resistance
    
    def _detect_bounces(self, price: float, high: float, low: float, 
                       nearest_support: Optional[float], nearest_resistance: Optional[float]) -> None:
        """Detect and count successful bounces from trendlines."""
        if self._prev_price is None or self._prev_low is None or self._prev_high is None:
            return
        
        bounce_threshold = self.tolerance * 2  # Slightly wider for bounce detection
        
        # Check for support bounce
        if nearest_support is not None:
            # Did previous bar touch/penetrate support?
            prev_touched_support = (
                self._prev_low <= nearest_support * (1 + bounce_threshold) and
                self._prev_price > nearest_support * (1 - bounce_threshold)
            )
            
            # Is current bar bouncing up from support?
            curr_bouncing_up = (
                low > nearest_support * (1 - bounce_threshold) and
                price > self._prev_price and
                price > nearest_support
            )
            
            # Detect successful bounce
            if prev_touched_support and curr_bouncing_up:
                # Avoid counting same bounce multiple times
                if self._bar_index - self._last_support_touch_bar > 3:
                    self._support_bounces += 1
                    self._last_support_touch_bar = self._bar_index
        
        # Check for resistance bounce
        if nearest_resistance is not None:
            # Did previous bar touch/penetrate resistance?
            prev_touched_resistance = (
                self._prev_high >= nearest_resistance * (1 - bounce_threshold) and
                self._prev_price < nearest_resistance * (1 + bounce_threshold)
            )
            
            # Is current bar bouncing down from resistance?
            curr_bouncing_down = (
                high < nearest_resistance * (1 + bounce_threshold) and
                price < self._prev_price and
                price < nearest_resistance
            )
            
            # Detect successful bounce
            if prev_touched_resistance and curr_bouncing_down:
                # Avoid counting same bounce multiple times
                if self._bar_index - self._last_resistance_touch_bar > 3:
                    self._resistance_bounces += 1
                    self._last_resistance_touch_bar = self._bar_index
    
    def _get_nearest_slopes(self, nearest_support: Optional[float], 
                           nearest_resistance: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """Get slopes of nearest support and resistance lines."""
        support_slope = None
        resistance_slope = None
        
        # Find slope of nearest support line
        if nearest_support is not None:
            for line in self._valid_uptrend_lines:
                if abs(line['current_val'] - nearest_support) < 0.001:
                    support_slope = line['slope']
                    break
        
        # Find slope of nearest resistance line
        if nearest_resistance is not None:
            for line in self._valid_downtrend_lines:
                if abs(line['current_val'] - nearest_resistance) < 0.001:
                    resistance_slope = -line['slope']  # Negative because downtrend
                    break
        
        return support_slope, resistance_slope
    
    def _detect_branches(self) -> Tuple[int, int]:
        """Detect branching patterns where steeper trendlines branch off from existing ones."""
        uptrend_branches = 0
        downtrend_branches = 0
        
        # Check uptrend branches (newer, steeper uptrends)
        if len(self._valid_uptrend_lines) >= 2:
            # Sort by slope (steepest first)
            sorted_uptrends = sorted(self._valid_uptrend_lines, key=lambda x: x['slope'], reverse=True)
            
            for i in range(len(sorted_uptrends) - 1):
                steeper = sorted_uptrends[i]
                flatter = sorted_uptrends[i + 1]
                
                # Check if steeper line starts near the flatter line
                # and has a significantly steeper slope
                if (steeper['start_idx'] > flatter['start_idx'] and
                    steeper['slope'] > flatter['slope'] * 1.2):  # 20% steeper
                    
                    # Check if they intersect or start close
                    steeper_start_val = steeper['start_val']
                    flatter_val_at_start = (flatter['slope'] * steeper['start_idx'] + 
                                          flatter['start_val'] - flatter['slope'] * flatter['start_idx'])
                    
                    if abs(steeper_start_val - flatter_val_at_start) / flatter_val_at_start < 0.01:
                        uptrend_branches += 1
        
        # Check downtrend branches (newer, steeper downtrends)
        if len(self._valid_downtrend_lines) >= 2:
            # Sort by slope (steepest descent first - most negative)
            sorted_downtrends = sorted(self._valid_downtrend_lines, key=lambda x: x['slope'])
            
            for i in range(len(sorted_downtrends) - 1):
                steeper = sorted_downtrends[i]
                flatter = sorted_downtrends[i + 1]
                
                # Check if steeper line starts near the flatter line
                if (steeper['start_idx'] > flatter['start_idx'] and
                    abs(steeper['slope']) > abs(flatter['slope']) * 1.2):  # 20% steeper descent
                    
                    # Check if they intersect or start close
                    steeper_start_val = steeper['start_val']
                    flatter_val_at_start = (flatter['start_val'] - 
                                          flatter['slope'] * (steeper['start_idx'] - flatter['start_idx']))
                    
                    if abs(steeper_start_val - flatter_val_at_start) / flatter_val_at_start < 0.01:
                        downtrend_branches += 1
        
        return uptrend_branches, downtrend_branches
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        self._bar_index = 0
        self._pivot_highs.clear()
        self._pivot_lows.clear()
        self._valid_uptrend_lines.clear()
        self._valid_downtrend_lines.clear()
        
        # Reset bounce tracking
        self._prev_price = None
        self._prev_low = None
        self._prev_high = None
        self._support_bounces = 0
        self._resistance_bounces = 0
        self._last_support_touch_bar = -1
        self._last_resistance_touch_bar = -1


class DiagonalChannel:
    """Diagonal Channel detection with parallel trendline tracking."""
    
    def __init__(self, lookback: int = 20, min_points: int = 3, 
                 channel_tolerance: float = 0.02, parallel_tolerance: float = 0.1,
                 name: str = "diagonal_channel"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self.min_points = min_points
        self.channel_tolerance = channel_tolerance
        self.parallel_tolerance = parallel_tolerance
        
        # Buffers
        self._high_buffer = deque(maxlen=lookback * 4)
        self._low_buffer = deque(maxlen=lookback * 4)
        self._close_buffer = deque(maxlen=lookback * 4)
        self._bar_index = 0
        
        # Pivot tracking
        self._pivot_highs = deque(maxlen=20)
        self._pivot_lows = deque(maxlen=20)
        
        # Channel tracking
        self._current_channel = None
        self._channel_touches = 0
        self._upper_bounces = 0
        self._lower_bounces = 0
        self._last_upper_touch_bar = -1
        self._last_lower_touch_bar = -1
        
        # Previous values for bounce detection
        self._prev_price = None
        self._prev_low = None
        self._prev_high = None
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if high is None or low is None:
            raise ValueError("DiagonalChannel requires high and low prices")
        
        self._bar_index += 1
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        self._close_buffer.append(price)
        
        # Detect pivot points
        self._detect_pivots()
        
        # Find best channel
        channel = self._find_best_channel()
        
        # Update current channel if found
        if channel:
            self._current_channel = channel
        
        # Use current or last valid channel
        active_channel = self._current_channel
        
        if active_channel:
            # Calculate channel values at current bar
            upper_value = self._get_channel_value(active_channel['upper'], self._bar_index)
            lower_value = self._get_channel_value(active_channel['lower'], self._bar_index)
            mid_value = (upper_value + lower_value) / 2
            width = upper_value - lower_value
            
            # Detect touches and bounces
            self._detect_channel_interactions(price, high, low, upper_value, lower_value)
            
            # Calculate position in channel (0 = lower, 1 = upper)
            position = (price - lower_value) / width if width > 0 else 0.5
            
            # Check if channel is still valid (for metadata)
            channel_age = self._bar_index - active_channel.get('last_valid_bar', self._bar_index)
            is_current = (channel == active_channel)
            
            self._state.set_value({
                "upper_channel": upper_value,
                "lower_channel": lower_value,
                "mid_channel": mid_value,
                "channel_width": width,
                "channel_angle": math.degrees(math.atan(active_channel['upper']['slope'])),
                "position_in_channel": position,
                "channel_touches": self._channel_touches,
                "upper_bounces": self._upper_bounces,
                "lower_bounces": self._lower_bounces,
                "bars_in_channel": active_channel.get('bars_valid', 0),
                "channel_strength": active_channel.get('strength', 0),
                "channel_is_current": is_current,
                "channel_age": channel_age
            })
        else:
            # Never had a valid channel
            self._state.set_value(None)
        
        # Update previous values
        self._prev_price = price
        self._prev_low = low
        self._prev_high = high
        
        return self._state.value
    
    def _detect_pivots(self):
        """Detect pivot highs and lows."""
        lookback = min(self.lookback, 3)
        
        if len(self._high_buffer) >= 2 * lookback + 1:
            buffer_len = len(self._high_buffer)
            mid_idx = buffer_len - lookback - 1
            
            if mid_idx >= lookback:
                mid_high = self._high_buffer[mid_idx]
                mid_low = self._low_buffer[mid_idx]
                
                # Check for pivot high
                is_pivot_high = all(
                    self._high_buffer[mid_idx - i] < mid_high and 
                    self._high_buffer[mid_idx + i] < mid_high
                    for i in range(1, lookback + 1)
                )
                
                if is_pivot_high:
                    pivot_bar_idx = self._bar_index - (buffer_len - 1 - mid_idx)
                    self._pivot_highs.append({
                        'bar': pivot_bar_idx,
                        'value': mid_high
                    })
                
                # Check for pivot low
                is_pivot_low = all(
                    self._low_buffer[mid_idx - i] > mid_low and 
                    self._low_buffer[mid_idx + i] > mid_low
                    for i in range(1, lookback + 1)
                )
                
                if is_pivot_low:
                    pivot_bar_idx = self._bar_index - (buffer_len - 1 - mid_idx)
                    self._pivot_lows.append({
                        'bar': pivot_bar_idx,
                        'value': mid_low
                    })
    
    def _find_best_channel(self) -> Optional[Dict[str, Any]]:
        """Find the best diagonal channel from pivot points."""
        if len(self._pivot_highs) < self.min_points or len(self._pivot_lows) < self.min_points:
            return None
        
        best_channel = None
        best_score = 0
        
        # Try different combinations of pivot highs for upper channel
        for i in range(len(self._pivot_highs) - 1):
            for j in range(i + 1, len(self._pivot_highs)):
                upper_line = self._create_line(self._pivot_highs[i], self._pivot_highs[j])
                
                # Find parallel lower line from pivot lows
                for k in range(len(self._pivot_lows) - 1):
                    for l in range(k + 1, len(self._pivot_lows)):
                        lower_line = self._create_line(self._pivot_lows[k], self._pivot_lows[l])
                        
                        # Check if lines are parallel
                        if self._are_parallel(upper_line, lower_line):
                            channel = self._validate_channel(upper_line, lower_line)
                            if channel and channel['score'] > best_score:
                                best_channel = channel
                                best_score = channel['score']
        
        return best_channel
    
    def _create_line(self, p1: Dict, p2: Dict) -> Dict[str, Any]:
        """Create a line from two points."""
        if p1['bar'] == p2['bar']:
            return None
        
        slope = (p2['value'] - p1['value']) / (p2['bar'] - p1['bar'])
        intercept = p1['value'] - slope * p1['bar']
        
        return {
            'slope': slope,
            'intercept': intercept,
            'start_bar': min(p1['bar'], p2['bar']),
            'end_bar': max(p1['bar'], p2['bar'])
        }
    
    def _are_parallel(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines are parallel within tolerance."""
        if not line1 or not line2:
            return False
        
        slope_diff = abs(line1['slope'] - line2['slope'])
        avg_slope = abs(line1['slope'] + line2['slope']) / 2
        
        if avg_slope > 0:
            return slope_diff / avg_slope < self.parallel_tolerance
        else:
            return slope_diff < 0.0001
    
    def _validate_channel(self, upper_line: Dict, lower_line: Dict) -> Optional[Dict[str, Any]]:
        """Validate a channel and calculate its score."""
        touches = 0
        violations = 0
        bars_valid = 0
        
        start_bar = max(upper_line['start_bar'], lower_line['start_bar'])
        end_bar = min(self._bar_index, 
                      max(upper_line['end_bar'], lower_line['end_bar']) + self.lookback)
        
        for i in range(max(0, len(self._close_buffer) - (self._bar_index - start_bar)), 
                      len(self._close_buffer)):
            bar_idx = self._bar_index - (len(self._close_buffer) - i - 1)
            if bar_idx < start_bar or bar_idx > end_bar:
                continue
            
            high = self._high_buffer[i]
            low = self._low_buffer[i]
            
            upper_val = self._get_channel_value(upper_line, bar_idx)
            lower_val = self._get_channel_value(lower_line, bar_idx)
            
            # Check if price respects channel
            if high > upper_val * (1 + self.channel_tolerance):
                violations += 1
            elif high >= upper_val * (1 - self.channel_tolerance):
                touches += 1
            
            if low < lower_val * (1 - self.channel_tolerance):
                violations += 1
            elif low <= lower_val * (1 + self.channel_tolerance):
                touches += 1
            
            bars_valid += 1
        
        if violations > touches / 2 or touches < self.min_points:
            return None
        
        # Calculate channel score
        width = self._get_channel_value(upper_line, self._bar_index) - \
                self._get_channel_value(lower_line, self._bar_index)
        
        score = touches * bars_valid / (violations + 1) * (1 - abs(upper_line['slope']) * 0.1)
        
        return {
            'upper': upper_line,
            'lower': lower_line,
            'touches': touches,
            'violations': violations,
            'bars_valid': bars_valid,
            'score': score,
            'strength': min(1.0, touches / (self.min_points * 2)),
            'last_valid_bar': self._bar_index
        }
    
    def _get_channel_value(self, line: Dict, bar: int) -> float:
        """Get channel line value at specific bar."""
        return line['slope'] * bar + line['intercept']
    
    def _detect_channel_interactions(self, price: float, high: float, low: float,
                                   upper: float, lower: float) -> None:
        """Detect touches and bounces with channel boundaries."""
        if self._prev_price is None:
            return
        
        touch_threshold = self.channel_tolerance
        
        # Check upper channel
        if high >= upper * (1 - touch_threshold):
            self._channel_touches += 1
            
            # Check for bounce
            if (self._prev_high >= upper * (1 - touch_threshold) and 
                price < upper and price < self._prev_price):
                if self._bar_index - self._last_upper_touch_bar > 3:
                    self._upper_bounces += 1
                    self._last_upper_touch_bar = self._bar_index
        
        # Check lower channel
        if low <= lower * (1 + touch_threshold):
            self._channel_touches += 1
            
            # Check for bounce
            if (self._prev_low <= lower * (1 + touch_threshold) and 
                price > lower and price > self._prev_price):
                if self._bar_index - self._last_lower_touch_bar > 3:
                    self._lower_bounces += 1
                    self._last_lower_touch_bar = self._bar_index
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        self._bar_index = 0
        self._pivot_highs.clear()
        self._pivot_lows.clear()
        self._current_channel = None
        self._channel_touches = 0
        self._upper_bounces = 0
        self._lower_bounces = 0
        self._last_upper_touch_bar = -1
        self._last_lower_touch_bar = -1
        self._prev_price = None
        self._prev_low = None
        self._prev_high = None


class PricePeaks:
    """Peak and trough detection for divergence analysis."""
    
    def __init__(self, lookback: int = 20, min_prominence: float = 0.001, name: str = "price_peaks"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self.min_prominence = min_prominence  # Minimum % change for valid peak/trough
        
        # Buffers
        self._high_buffer = deque(maxlen=lookback * 3)
        self._low_buffer = deque(maxlen=lookback * 3)
        self._close_buffer = deque(maxlen=lookback * 3)
        
        # Peak/trough tracking
        self._last_peak_value = None
        self._last_peak_bar = -1
        self._last_trough_value = None
        self._last_trough_bar = -1
        self._bar_index = 0
        
        # Current detection state
        self._is_peak = False
        self._is_trough = False
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if high is None or low is None:
            raise ValueError("Peak detection requires high and low prices")
        
        self._bar_index += 1
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        self._close_buffer.append(price)
        
        # Reset current detection flags
        self._is_peak = False
        self._is_trough = False
        
        # Need enough data for detection
        if len(self._high_buffer) < self.lookback * 2 + 1:
            return None
        
        # Check for peak/trough at the middle of the buffer
        mid_idx = self.lookback
        mid_high = self._high_buffer[mid_idx]
        mid_low = self._low_buffer[mid_idx]
        mid_close = self._close_buffer[mid_idx]
        
        # Peak detection
        left_max = max(self._high_buffer[i] for i in range(mid_idx))
        right_max = max(self._high_buffer[i] for i in range(mid_idx + 1, len(self._high_buffer)))
        
        if mid_high > left_max and mid_high > right_max:
            # Check prominence
            prominence = (mid_high - min(left_max, right_max)) / mid_high
            if prominence > self.min_prominence:
                self._is_peak = True
                self._last_peak_value = mid_high
                self._last_peak_bar = self._bar_index - self.lookback
        
        # Trough detection
        left_min = min(self._low_buffer[i] for i in range(mid_idx))
        right_min = min(self._low_buffer[i] for i in range(mid_idx + 1, len(self._low_buffer)))
        
        if mid_low < left_min and mid_low < right_min:
            # Check prominence
            prominence = (max(left_min, right_min) - mid_low) / mid_low
            if prominence > self.min_prominence:
                self._is_trough = True
                self._last_trough_value = mid_low
                self._last_trough_bar = self._bar_index - self.lookback
        
        # Calculate bars since last peak/trough
        bars_since_peak = self._bar_index - self._last_peak_bar if self._last_peak_bar > 0 else -1
        bars_since_trough = self._bar_index - self._last_trough_bar if self._last_trough_bar > 0 else -1
        
        self._state.set_value({
            'is_peak': self._is_peak,
            'is_trough': self._is_trough,
            'last_peak_value': self._last_peak_value,
            'last_trough_value': self._last_trough_value,
            'bars_since_last_peak': bars_since_peak,
            'bars_since_last_trough': bars_since_trough,
            'current_price': price
        })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._close_buffer.clear()
        self._last_peak_value = None
        self._last_peak_bar = -1
        self._last_trough_value = None
        self._last_trough_bar = -1
        self._bar_index = 0
        self._is_peak = False
        self._is_trough = False


# Structure feature registry for the FeatureHub factory
STRUCTURE_FEATURES = {
    "pivot_points": PivotPoints,
    "support_resistance": SupportResistance,
    "sr": SupportResistance,  # Alias
    "swing_points": SwingPoints,
    "swing": SwingPoints,  # Alias
    "linear_regression": LinearRegression,
    "linreg": LinearRegression,  # Alias
    "fibonacci_retracement": FibonacciRetracement,
    "fibonacci": FibonacciRetracement,  # Alias
    "fib": FibonacciRetracement,  # Alias
    "trendlines": TrendLines,
    "trendline": TrendLines,  # Alias
    "diagonal_channel": DiagonalChannel,
    "channel": DiagonalChannel,  # Alias
    "price_peaks": PricePeaks,
    "peaks": PricePeaks,  # Alias
}