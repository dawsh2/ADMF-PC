"""
Bollinger RSI Divergence using self-contained feature
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_self_contained',
    feature_discovery=lambda params: [
        FeatureSpec(feature_type='bb_rsi_divergence_self', params={})
    ],
    parameter_space={},
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'divergence', 'mean_reversion', 'self_contained']
)
def bollinger_rsi_self_contained(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trade RSI divergences using self-contained feature.
    """
    
    # Get the divergence signal
    divergence = features.get('bb_rsi_divergence_self', {})
    signal_value = divergence.get('value', 0)
    
    # Debug logging
    import sys
    if signal_value != 0:
        print(f"[BB_RSI_SELF] Bar {bar.get('bar_index', 'unknown')}: signal={signal_value}, divergence={divergence}", file=sys.stderr)
    
    # Build response
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_rsi_self_contained',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'reason': divergence.get('reason', ''),
            'stage': divergence.get('stage', ''),
            'signals_generated': divergence.get('signals_generated', 0),
            'bar_count': divergence.get('bar_count', 0),
            'close': bar.get('close', 0)
        }
    }