#!/usr/bin/env python3
"""
Check how main.py executes strategies with filters.
"""

import subprocess
import sys

def check_main_execution():
    print("=== CHECKING MAIN.PY EXECUTION ===\n")
    
    # Check main.py help for filter-related options
    print("1. CHECKING MAIN.PY OPTIONS:")
    print("-" * 60)
    
    try:
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True)
        help_text = result.stdout
        
        # Look for filter-related options
        filter_lines = [line for line in help_text.split('\n') 
                       if 'filter' in line.lower()]
        
        if filter_lines:
            print("Filter-related options found:")
            for line in filter_lines:
                print(f"  {line.strip()}")
        else:
            print("No filter-related options found in help")
            
    except Exception as e:
        print(f"Error running main.py --help: {e}")
    
    # Check if there's a specific flag needed
    print("\n2. COMMON EXECUTION PATTERNS:")
    print("-" * 60)
    
    print("Standard execution:")
    print("  python main.py config.yaml")
    print("\nWith debug:")
    print("  python main.py config.yaml --debug")
    print("\nWith specific data:")
    print("  python main.py config.yaml --data path/to/data")
    
    # Look for filter-related code in main.py
    print("\n3. CHECKING MAIN.PY FOR FILTER HANDLING:")
    print("-" * 60)
    
    try:
        with open("main.py", 'r') as f:
            main_content = f.read()
            
        # Check for filter-related imports or code
        if 'filter' in main_content.lower():
            print("Found filter-related code in main.py")
            
            # Extract relevant lines
            lines = main_content.split('\n')
            for i, line in enumerate(lines):
                if 'filter' in line.lower() and not line.strip().startswith('#'):
                    print(f"  Line {i+1}: {line.strip()}")
        else:
            print("No filter-related code found in main.py")
            
    except Exception as e:
        print(f"Error reading main.py: {e}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS:")
    print("="*60)
    print("The filter might need to be:")
    print("1. Enabled with a specific flag")
    print("2. Applied at a different stage in the pipeline")
    print("3. Implemented differently than expected")
    print("\nThe fact that the main config works suggests the issue")
    print("is specific to our config format or execution method.")

if __name__ == "__main__":
    check_main_execution()