# How to Completely Exit Python

## 1. From Python Interactive Shell (>>>)
```python
# Method 1: Use exit function
exit()

# Method 2: Use quit function  
quit()

# Method 3: Press Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows)

# Method 4: Type EOF
raise SystemExit
```

## 2. From IPython
```python
# Method 1
exit

# Method 2
quit

# Method 3: Press Ctrl+D twice
```

## 3. From Jupyter Notebook
- Close the browser tab
- In terminal where jupyter is running: Press Ctrl+C twice
- Or: File → Shut Down

## 4. From VS Code Python Terminal
- Click the trash can icon in the terminal
- Or close the terminal panel
- Or press Ctrl+C then close terminal

## 5. From Command Line Script
If running `python main.py`:
- Press Ctrl+C to interrupt
- This returns you to bash/terminal

## 6. Check if Python is Really Closed
```bash
# Check if any Python processes are running
ps aux | grep python

# Kill all Python processes (careful!)
pkill python
```

## After Exiting Python

1. Clear the bytecode cache:
```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
```

2. Start fresh:
```bash
python main.py --config config/bollinger/test.yaml
```

## The Real Issue: Missing OHLC Data

You've identified the critical problem! If signals only have close price, the risk manager can't:
- Use the LOW price to check if stop loss was hit intrabar
- Use the HIGH price to check if take profit was hit intrabar

This would cause exits at wrong prices!