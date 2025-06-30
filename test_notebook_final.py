#!/usr/bin/env python3
"""Test complete notebook launch flow"""

from pathlib import Path
from src.analytics.papermill_runner import PapermillNotebookRunner
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_complete_launch():
    """Test complete notebook launch flow"""
    results_dir = "/Users/daws/ADMF-PC/config/bollinger/results/20250629_155112"
    
    # Initialize runner
    runner = PapermillNotebookRunner()
    
    # Config for notebook
    config = {
        'name': 'bollinger_analysis',
        'data': 'SPY_5m',
    }
    
    print(f"Testing complete notebook launch for: {results_dir}")
    
    # Test with execute=True but launch=False to see if it works
    notebook_path = runner.run_analysis(
        run_dir=Path(results_dir),
        config=config,
        execute=True,
        launch=False
    )
    
    if notebook_path:
        print(f"✅ Notebook executed successfully: {notebook_path}")
        print(f"\nTo view the notebook:")
        print(f"  jupyter lab {notebook_path}")
    else:
        print("❌ Failed to execute notebook")

if __name__ == "__main__":
    test_complete_launch()