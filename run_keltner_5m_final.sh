#!/bin/bash
# Run Keltner Bands on 5-minute data

echo "Running Keltner Bands optimization on 5-minute data..."
echo "Configuration: 5 periods x 5 multipliers = 25 strategies"
echo ""

# Run the signal generation
python3 main.py \
    --config config/indicators/volatility/test_keltner_bands_5m_optimized.yaml \
    --signal-generation \
    --dataset SPY_5m

echo ""
echo "If successful, look for the workspace path above"
echo "Then run: python3 analyze_keltner_5m.py workspaces/signal_generation_XXXXXXXX"