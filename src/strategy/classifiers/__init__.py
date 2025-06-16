"""
Market classification components for Step 3.

This module provides classifiers and containers for regime detection
and adaptive trading based on market conditions.

Now includes pure function classifiers decorated with @classifier
for automatic discovery.
"""

# Step 3 Components
from .regime_types import (
    RegimeChangeEvent,
    RegimeState,
    ClassificationFeatures,
    ClassifierConfig
)

# Import pure function classifiers for discovery
from .classifiers import (
    MarketRegime,
    trend_classifier,
    volatility_classifier,
    momentum_regime_classifier
)

# Import additional classifiers
from .market_state_classifier import market_state_classifier

# Import multi-state classifiers (renamed from enhanced_multi_state_classifiers)
from .multi_state_classifiers import (
    multi_timeframe_trend_classifier,
    volatility_momentum_classifier,
    market_regime_classifier,
    microstructure_classifier,
    hidden_markov_classifier
)

# Import intraday ORB classifiers
from .intraday_orb_classifiers import (
    intraday_orb_classifier,
    microstructure_momentum_classifier,
    session_pattern_classifier
)

# Old class-based classifiers moved to tmp/deprecated_classifiers/
# Now using pure function classifiers with @classifier decorator

__all__ = [
    # Types and Events
    'MarketRegime',
    'RegimeChangeEvent', 
    'RegimeState',
    'ClassificationFeatures',
    'ClassifierConfig',
    
    # Pure function classifiers (decorated with @classifier)
    'trend_classifier',
    'volatility_classifier',
    'momentum_regime_classifier',
    'market_state_classifier',
    
    # Multi-state classifiers
    'multi_timeframe_trend_classifier',
    'volatility_momentum_classifier',
    'market_regime_classifier',
    'microstructure_classifier',
    'hidden_markov_classifier',
    
    # Intraday ORB classifiers
    'intraday_orb_classifier',
    'microstructure_momentum_classifier',
    'session_pattern_classifier'
]