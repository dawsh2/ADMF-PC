"""
Indicator-based Strategy Implementations.

Atomic building blocks that can be composed into ensemble strategies.
Each indicator strategy is stateless and generates binary signals.

Consolidated into three logical files:
- crossovers.py: All crossover-based strategies (9 strategies)
- oscillators.py: All oscillator-based strategies (4 strategies) 
- volatility.py: All volatility-based strategies (3 strategies)
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
    'bollinger_breakout'
]