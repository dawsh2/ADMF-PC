#!/bin/bash
# Regime-Adaptive Trading System Pipeline
# 
# This example shows how to build a complete regime-adaptive trading system
# using Unix pipes to chain phases together.

set -eo pipefail  # Exit on error, fail on pipe errors

# Configuration files
GRID_SEARCH_CONFIG="config/regime_adaptive/grid_search.yaml"
VALIDATION_DATA="config/regime_adaptive/validation.yaml"

# Output directory
OUTPUT_DIR="results/regime_adaptive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Regime-Adaptive Trading System Pipeline"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo

# Phase 1: Grid Search Signal Generation
echo "Phase 1: Generating signals with parameter grid search..."
echo "--------------------------------------------------"

python main.py --signal-generation "$GRID_SEARCH_CONFIG" \
    --output-format pipe \
    --verbose | tee "$OUTPUT_DIR/phase1_signals.json" | \

# Phase 2: Regime-Based Ensemble Optimization  
(
    echo
    echo "Phase 2: Optimizing ensemble weights per regime..."
    echo "--------------------------------------------------"
    python main.py --signal-replay --from-pipe \
        --optimize-ensemble \
        --output-format pipe \
        --verbose
) | tee "$OUTPUT_DIR/phase2_ensemble.json" | \

# Phase 3: Final Validation
(
    echo
    echo "Phase 3: Running final validation backtest..."
    echo "--------------------------------------------------"
    python main.py --backtest --from-pipe \
        --validate \
        --output-format human \
        --verbose
) | tee "$OUTPUT_DIR/phase3_results.txt"

echo
echo "=================================================="
echo "Pipeline Complete!"
echo "=================================================="
echo "Results saved to: $OUTPUT_DIR"
echo
echo "Files generated:"
ls -la "$OUTPUT_DIR"

# Optional: Generate summary report
echo
echo "Generating summary report..."
jq -s '{
    signal_generation: .[0],
    ensemble_optimization: .[1],
    validation_metrics: {
        sharpe_ratio: .[1].results.optimal_sharpe,
        total_signals: .[0].results.total_signals,
        regimes: .[1].results.regimes_identified
    }
}' "$OUTPUT_DIR/phase1_signals.json" \
    "$OUTPUT_DIR/phase2_ensemble.json" > "$OUTPUT_DIR/summary.json"

echo "Summary saved to: $OUTPUT_DIR/summary.json"

# Example parallel variant:
# Run multiple strategies in parallel, then combine
parallel_example() {
    echo
    echo "Parallel Example: Running 4 strategies concurrently..."
    echo "--------------------------------------------------"
    
    parallel -j 4 --tag '
        python main.py --signal-generation {} --output-format pipe
    ' ::: config/strategies/momentum_*.yaml | \
    python main.py --signal-replay --from-pipe --merge --optimize-ensemble
}

# Conditional execution example:
conditional_example() {
    echo
    echo "Conditional Example: Only optimize if enough signals..."
    echo "--------------------------------------------------"
    
    python main.py --signal-generation "$GRID_SEARCH_CONFIG" --output-format pipe | \
    tee phase1.json | \
    jq -e '.results.total_signals > 10000' && \
    python main.py --signal-replay --from-pipe --optimize-ensemble || \
    echo "Not enough signals generated, skipping optimization"
}

# Real-time monitoring example:
monitoring_example() {
    echo
    echo "Monitoring Example: Watch progress in real-time..."
    echo "--------------------------------------------------"
    
    # Use pv (pipe viewer) to monitor data flow
    python main.py --signal-generation "$GRID_SEARCH_CONFIG" --output-format pipe | \
    pv -l -N "Signals" | \
    python main.py --signal-replay --from-pipe --optimize-ensemble --output-format pipe | \
    pv -l -N "Ensemble" | \
    python main.py --backtest --from-pipe --validate
}