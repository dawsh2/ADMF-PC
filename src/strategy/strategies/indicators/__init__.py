"""
Indicator-based Strategy Implementations.

Atomic building blocks that can be composed into ensemble strategies.
Each indicator strategy is stateless and generates binary signals.

Organized by indicator category:
- crossovers.py: All crossover-based strategies (9 strategies)
- oscillators.py: All oscillator-based strategies (4 strategies) 
- volatility.py: All volatility-based strategies (3 strategies)
- trend.py: All trend-based strategies 
- volume.py: All volume-based strategies
- structure.py: All structure-based strategies
- momentum.py: All momentum-based strategies (7 strategies)
"""

# Import crossover strategies
from .crossovers import (
    sma_crossover,
    ema_sma_crossover,
    ema_crossover,
    dema_sma_crossover,
    dema_crossover,
    tema_sma_crossover,
    stochastic_crossover,
    vortex_crossover,
    ichimoku_cloud_position
)

# Import oscillator strategies  
from .oscillators import (
    rsi_threshold,
    rsi_bands,
    cci_threshold,
    cci_bands
)

# Import volatility strategies
from .volatility import (
    keltner_breakout,
    donchian_breakout,
    bollinger_breakout
)

# Import momentum strategies
from .momentum import (
    macd_crossover_strategy,
    momentum_breakout_strategy,
    roc_trend_strategy,
    adx_trend_strength_strategy,
    aroon_oscillator_strategy,
    vortex_trend_strategy,
    momentum_composite_strategy
)

__all__ = [
    # Crossover strategies (9 total)
    'sma_crossover',
    'ema_sma_crossover', 
    'ema_crossover',
    'dema_sma_crossover',
    'dema_crossover',
    'tema_sma_crossover',
    'stochastic_crossover',
    'vortex_crossover',
    'ichimoku_cloud_position',
    
    # Oscillator strategies (4 total)
    'rsi_threshold',
    'rsi_bands',
    'cci_threshold',
    'cci_bands',
    
    # Volatility strategies (3 total)
    'keltner_breakout',
    'donchian_breakout',
    'bollinger_breakout',
    
    # Momentum strategies (7 total)
    'macd_crossover_strategy',
    'momentum_breakout_strategy',
    'roc_trend_strategy',
    'adx_trend_strength_strategy',
    'aroon_oscillator_strategy',
    'vortex_trend_strategy',
    'momentum_composite_strategy'
]