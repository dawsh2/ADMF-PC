#!/bin/bash
# Load Alpaca environment variables from .zshrc

# Source the .zshrc to get the environment variables
source ~/.zshrc

# Run the python script with all arguments passed through
python "$@"