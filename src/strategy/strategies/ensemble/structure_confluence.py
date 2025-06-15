"""
Structure Confluence Ensemble Strategies.

Composite strategies that combine multiple structure indicators
(pivots, trendlines, S/R levels) to identify high-probability setups.
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='pivot_trendline_confluence',
    feature_config=['pivot_channels', 'trendlines', 'sr_confluence']
)
def pivot_trendline_confluence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pivot-Trendline Confluence strategy.
    
    Generates signals when both pivot levels and trendlines align,
    creating high-probability reversal zones.
    
    Logic:
    - Long: Price bounces from BOTH pivot support AND uptrend line
    - Short: Price bounces from BOTH pivot resistance AND downtrend line
    - Strength based on confluence score
    """
    min_confluence = params.get('min_confluence', 2)
    proximity_threshold = params.get('proximity_threshold', 0.003)  # 0.3%
    
    # Get pivot data
    pivot_high = features.get('pivot_channels_pivot_high')
    pivot_low = features.get('pivot_channels_pivot_low')
    pivot_bounce_up = features.get('pivot_channels_bounce_up', False)
    pivot_bounce_down = features.get('pivot_channels_bounce_down', False)
    
    # Get trendline data
    nearest_support = features.get('trendlines_nearest_support')
    nearest_resistance = features.get('trendlines_nearest_resistance')
    trendline_bounces = features.get('trendlines_recent_bounces', [])
    
    # Get confluence data
    strongest_support = features.get('sr_confluence_strongest_support')
    strongest_resistance = features.get('sr_confluence_strongest_resistance')
    max_support_confluence = features.get('sr_confluence_max_support_confluence', 0)
    max_resistance_confluence = features.get('sr_confluence_max_resistance_confluence', 0)
    
    price = bar.get('close', 0)
    
    # Check for confluence zones
    support_confluence = 0
    resistance_confluence = 0
    
    # Support confluence check
    if pivot_low and nearest_support:
        if abs(pivot_low - nearest_support) / pivot_low <= proximity_threshold:
            support_confluence += 2  # Pivot + Trendline alignment
    
    # Resistance confluence check  
    if pivot_high and nearest_resistance:
        if abs(pivot_high - nearest_resistance) / pivot_high <= proximity_threshold:
            resistance_confluence += 2  # Pivot + Trendline alignment
    
    # Add general confluence scores
    support_confluence += max_support_confluence
    resistance_confluence += max_resistance_confluence
    
    # Determine signal
    signal_value = 0
    confluence_score = 0
    
    # Check for bounces with confluence
    if pivot_bounce_up and support_confluence >= min_confluence:
        # Check if trendline also shows bounce
        uptrend_bounce = any(b['type'] == 'uptrend' for b in trendline_bounces)
        if uptrend_bounce:
            signal_value = 1
            confluence_score = support_confluence + 1  # Extra point for double bounce
        elif support_confluence >= min_confluence + 1:
            signal_value = 1
            confluence_score = support_confluence
    
    elif pivot_bounce_down and resistance_confluence >= min_confluence:
        # Check if trendline also shows bounce
        downtrend_bounce = any(b['type'] == 'downtrend' for b in trendline_bounces)
        if downtrend_bounce:
            signal_value = -1
            confluence_score = resistance_confluence + 1  # Extra point for double bounce
        elif resistance_confluence >= min_confluence + 1:
            signal_value = -1
            confluence_score = resistance_confluence
    
    # Proximity-based anticipatory signals for strong confluence
    elif support_confluence >= min_confluence + 1 and pivot_low:
        support_distance = (price - pivot_low) / price
        if 0 < support_distance < proximity_threshold:
            signal_value = 1  # Anticipatory long
            confluence_score = support_confluence
    
    elif resistance_confluence >= min_confluence + 1 and pivot_high:
        resistance_distance = (pivot_high - price) / price
        if 0 < resistance_distance < proximity_threshold:
            signal_value = -1  # Anticipatory short
            confluence_score = resistance_confluence
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'pivot_trendline_confluence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_confluence': min_confluence,
            'proximity_threshold': proximity_threshold,
            'price': price,
            'support_confluence': support_confluence,
            'resistance_confluence': resistance_confluence,
            'confluence_score': confluence_score,
            'pivot_high': pivot_high,
            'pivot_low': pivot_low,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
    }


@strategy(
    name='multi_touch_validation',
    feature_config=['pivot_channels', 'trendlines']
)
def multi_touch_validation(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Multi-Touch Validation strategy.
    
    Only trades S/R levels that have been validated by multiple touches.
    Higher touch count = higher confidence.
    
    Logic:
    - Requires minimum touches on both pivot and trendline
    - Weights recent touches more heavily
    - Combines touch counts for signal strength
    """
    min_pivot_touches = params.get('min_pivot_touches', 2)
    min_trendline_touches = params.get('min_trendline_touches', 3)
    recent_weight = params.get('recent_weight', 1.5)
    
    # Get pivot touch counts
    upper_touches = features.get('pivot_channels_upper_touches', 0)
    lower_touches = features.get('pivot_channels_lower_touches', 0)
    pivot_high_touches = features.get('pivot_channels_pivot_high_touches', 0)
    pivot_low_touches = features.get('pivot_channels_pivot_low_touches', 0)
    
    # Get pivot bounces
    pivot_bounce_up = features.get('pivot_channels_bounce_up', False)
    pivot_bounce_down = features.get('pivot_channels_bounce_down', False)
    
    # Get trendline data
    trendline_bounces = features.get('trendlines_recent_bounces', [])
    valid_uptrends = features.get('trendlines_valid_uptrends', 0)
    valid_downtrends = features.get('trendlines_valid_downtrends', 0)
    
    price = bar.get('close', 0)
    
    # Calculate total touches for support/resistance
    support_touches = max(lower_touches, pivot_low_touches)
    resistance_touches = max(upper_touches, pivot_high_touches)
    
    # Add trendline touches
    for bounce in trendline_bounces:
        if bounce['type'] == 'uptrend':
            support_touches += bounce['touches'] * recent_weight
        elif bounce['type'] == 'downtrend':
            resistance_touches += bounce['touches'] * recent_weight
    
    # Determine signal based on validated levels
    signal_value = 0
    touch_score = 0
    
    # Support bounce with validation
    if pivot_bounce_up and support_touches >= min_pivot_touches:
        # Check for trendline validation
        uptrend_validated = any(
            b['type'] == 'uptrend' and b['touches'] >= min_trendline_touches 
            for b in trendline_bounces
        )
        
        if uptrend_validated or support_touches >= min_pivot_touches + min_trendline_touches:
            signal_value = 1
            touch_score = support_touches
    
    # Resistance bounce with validation
    elif pivot_bounce_down and resistance_touches >= min_pivot_touches:
        # Check for trendline validation
        downtrend_validated = any(
            b['type'] == 'downtrend' and b['touches'] >= min_trendline_touches 
            for b in trendline_bounces
        )
        
        if downtrend_validated or resistance_touches >= min_pivot_touches + min_trendline_touches:
            signal_value = -1
            touch_score = resistance_touches
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'multi_touch_validation',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_pivot_touches': min_pivot_touches,
            'min_trendline_touches': min_trendline_touches,
            'recent_weight': recent_weight,
            'price': price,
            'support_touches': support_touches,
            'resistance_touches': resistance_touches,
            'touch_score': touch_score,
            'valid_uptrends': valid_uptrends,
            'valid_downtrends': valid_downtrends
        }
    }


@strategy(
    name='confluence_bounce_zones',
    feature_config=['sr_confluence', 'pivot_channels', 'trendlines']
)
def confluence_bounce_zones(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Confluence Bounce Zones strategy.
    
    Identifies and trades zones where multiple S/R sources converge,
    creating high-probability reversal areas.
    
    Logic:
    - Identifies confluence zones from multiple sources
    - Requires minimum confluence score
    - Trades bounces from these zones
    - Higher confluence = stronger signal
    """
    min_confluence_score = params.get('min_confluence_score', 3)
    zone_width = params.get('zone_width', 0.005)  # 0.5% zone width
    
    # Get confluence data
    support_zones = features.get('sr_confluence_support_zones', 0)
    resistance_zones = features.get('sr_confluence_resistance_zones', 0)
    strongest_support = features.get('sr_confluence_strongest_support')
    strongest_resistance = features.get('sr_confluence_strongest_resistance')
    max_support_confluence = features.get('sr_confluence_max_support_confluence', 0)
    max_resistance_confluence = features.get('sr_confluence_max_resistance_confluence', 0)
    active_bounces = features.get('sr_confluence_active_bounces', [])
    
    # Get additional structure data
    pivot_high = features.get('pivot_channels_pivot_high')
    pivot_low = features.get('pivot_channels_pivot_low')
    nearest_support = features.get('trendlines_nearest_support')
    nearest_resistance = features.get('trendlines_nearest_resistance')
    
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    
    # Check if we're in a confluence zone
    in_support_zone = False
    in_resistance_zone = False
    zone_strength = 0
    
    if strongest_support:
        support_distance = abs(price - strongest_support) / strongest_support
        if support_distance <= zone_width:
            in_support_zone = True
            zone_strength = max_support_confluence
    
    if strongest_resistance:
        resistance_distance = abs(price - strongest_resistance) / strongest_resistance
        if resistance_distance <= zone_width:
            in_resistance_zone = True
            zone_strength = max_resistance_confluence
    
    # Determine signal
    signal_value = 0
    
    # Check for active bounces in strong zones
    if active_bounces and zone_strength >= min_confluence_score:
        last_bounce = active_bounces[-1]
        
        if last_bounce['type'] == 'support_bounce':
            signal_value = 1
        elif last_bounce['type'] == 'resistance_bounce':
            signal_value = -1
    
    # Zone entry signals
    elif in_support_zone and zone_strength >= min_confluence_score:
        # Check if we're approaching from above
        if low <= strongest_support * (1 + zone_width) and price > strongest_support:
            signal_value = 1  # Support zone bounce
    
    elif in_resistance_zone and zone_strength >= min_confluence_score:
        # Check if we're approaching from below
        if high >= strongest_resistance * (1 - zone_width) and price < strongest_resistance:
            signal_value = -1  # Resistance zone bounce
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'confluence_bounce_zones',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_confluence_score': min_confluence_score,
            'zone_width': zone_width,
            'price': price,
            'support_zones': support_zones,
            'resistance_zones': resistance_zones,
            'zone_strength': zone_strength,
            'in_support_zone': in_support_zone,
            'in_resistance_zone': in_resistance_zone,
            'strongest_support': strongest_support,
            'strongest_resistance': strongest_resistance
        }
    }


@strategy(
    name='breakout_confluence',
    feature_config=['pivot_channels', 'trendlines', 'sr_confluence']
)
def breakout_confluence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Breakout Confluence strategy.
    
    Trades breakouts through multiple S/R levels simultaneously,
    indicating strong momentum.
    
    Logic:
    - Long: Breaks above multiple resistance levels
    - Short: Breaks below multiple support levels
    - Strength based on number of levels broken
    """
    min_levels_broken = params.get('min_levels_broken', 2)
    breakout_confirmation = params.get('breakout_confirmation', 0.002)  # 0.2% beyond level
    
    # Get breakout signals
    pivot_break_up = features.get('pivot_channels_break_up', False)
    pivot_break_down = features.get('pivot_channels_break_down', False)
    
    # Get trendline breaks
    trendline_breaks = features.get('trendlines_recent_breaks', [])
    
    # Get confluence breaks
    confluence_breaks = features.get('sr_confluence_recent_breaks', [])
    
    # Get key levels
    pivot_high = features.get('pivot_channels_pivot_high')
    pivot_low = features.get('pivot_channels_pivot_low')
    upper_channel = features.get('pivot_channels_upper_channel')
    lower_channel = features.get('pivot_channels_lower_channel')
    
    price = bar.get('close', 0)
    
    # Count breakout levels
    upside_breaks = 0
    downside_breaks = 0
    
    # Pivot breaks
    if pivot_break_up:
        upside_breaks += 1
    if pivot_break_down:
        downside_breaks += 1
    
    # Trendline breaks
    for tbreak in trendline_breaks:
        if tbreak['type'] == 'downtrend':  # Breaking above downtrend
            upside_breaks += 1
        elif tbreak['type'] == 'uptrend':  # Breaking below uptrend
            downside_breaks += 1
    
    # Confluence breaks
    for cbreak in confluence_breaks:
        if cbreak['type'] == 'resistance_break':
            upside_breaks += 1
        elif cbreak['type'] == 'support_break':
            downside_breaks += 1
    
    # Additional confirmation checks
    if pivot_high and price > pivot_high * (1 + breakout_confirmation):
        upside_breaks += 0.5  # Half point for confirmation
    if pivot_low and price < pivot_low * (1 - breakout_confirmation):
        downside_breaks += 0.5  # Half point for confirmation
    
    # Determine signal
    signal_value = 0
    breakout_strength = 0
    
    if upside_breaks >= min_levels_broken:
        signal_value = 1
        breakout_strength = upside_breaks
    elif downside_breaks >= min_levels_broken:
        signal_value = -1
        breakout_strength = downside_breaks
    
    # Always return current signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'breakout_confluence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'min_levels_broken': min_levels_broken,
            'breakout_confirmation': breakout_confirmation,
            'price': price,
            'upside_breaks': upside_breaks,
            'downside_breaks': downside_breaks,
            'breakout_strength': breakout_strength,
            'pivot_high': pivot_high,
            'pivot_low': pivot_low
        }
    }