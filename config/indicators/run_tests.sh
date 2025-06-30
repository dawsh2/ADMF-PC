#!/bin/bash
# Run indicator strategy tests

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Running Indicator Strategy Tests"
echo "================================"

# Function to run a test
run_test() {
    local config=$1
    local name=$(basename $config .yaml)
    
    echo -n "Testing $name... "
    
    if python main.py --config $config --signal-generation --bars 50 --dataset test > /tmp/${name}_output.log 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Error output:"
        tail -5 /tmp/${name}_output.log | sed 's/^/    /'
    fi
}

# Test crossover strategies
echo -e "\nCrossover Strategies:"
for config in config/indicators/crossover/*.yaml; do
    [ -f "$config" ] && run_test "$config"
done

# Test oscillator strategies
echo -e "\nOscillator Strategies:"
for config in config/indicators/oscillator/*.yaml; do
    [ -f "$config" ] && run_test "$config"
done

# Test volatility strategies
echo -e "\nVolatility Strategies:"
for config in config/indicators/volatility/*.yaml; do
    [ -f "$config" ] && run_test "$config"
done

# Test volume strategies
echo -e "\nVolume Strategies:"
for config in config/indicators/volume/*.yaml; do
    [ -f "$config" ] && run_test "$config"
done

echo -e "\nTest run complete!"