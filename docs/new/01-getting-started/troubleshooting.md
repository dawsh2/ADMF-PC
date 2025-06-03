# Troubleshooting Guide

This guide covers common issues you might encounter when setting up and using ADMF-PC, along with their solutions.

## 🚨 Installation Issues

### Python Version Problems

**Issue**: "Python version 3.11 or higher required"
```bash
python --version  # Shows 3.10 or lower
```

**Solution**:
```bash
# Install Python 3.11+ using pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0

# Or use system package manager
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
```

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the project root
pwd  # Should show /path/to/ADMF-PC

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or create .env file
echo "PYTHONPATH=$(pwd)" > .env
```

**Issue**: `ImportError: attempted relative import with no known parent package`

**Solution**:
```bash
# Always run from project root
cd /path/to/ADMF-PC
python main.py config/simple_backtest.yaml

# Don't run from subdirectories like:
# cd src/
# python ../main.py  # ❌ Wrong
```

### Permission Errors

**Issue**: Permission denied when installing packages
```bash
pip install -r requirements.txt
# ERROR: Could not install packages due to an EnvironmentError
```

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Or install with --user flag
pip install --user -r requirements.txt
```

## 📊 Data Issues

### Missing Data Files

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/SPY_1m.csv'`

**Solution**:
```bash
# Download sample data
python scripts/download_sample_data.py

# Or manually place your CSV files in data/ directory
mkdir -p data/
# Copy your CSV files here
```

### Invalid Data Format

**Issue**: `ValueError: Unable to parse date column`

**Solution**: Ensure your CSV has the correct format:
```csv
timestamp,open,high,low,close,volume
2023-01-01 09:30:00,400.0,401.5,399.5,401.0,1000000
2023-01-01 09:31:00,401.0,402.0,400.5,401.5,950000
```

Check common issues:
- Date format must be `YYYY-MM-DD HH:MM:SS`
- Column names must be lowercase
- No missing values in price data
- Volume must be integer

### Large Data Files

**Issue**: `MemoryError` when loading large datasets

**Solution**:
```yaml
# In your config file, use data chunking
data:
  source:
    type: "csv"
    path: "data/large_file.csv"
    chunk_size: 10000  # Process in chunks
    
# Or limit date range
data:
  start_date: "2023-01-01"
  end_date: "2023-03-31"  # 3 months instead of full year
```

## ⚙️ Configuration Issues

### YAML Syntax Errors

**Issue**: `yaml.scanner.ScannerError: mapping values are not allowed here`

**Solution**: Check YAML syntax:
```yaml
# ❌ Wrong - missing space after colon
workflow:
  type:"backtest"
  
# ✅ Correct
workflow:
  type: "backtest"
  
# ❌ Wrong - inconsistent indentation
strategies:
  - type: "momentum"
     params:  # Wrong indentation
       fast_period: 10
       
# ✅ Correct - consistent 2-space indentation
strategies:
  - type: "momentum"
    params:
      fast_period: 10
```

**Tip**: Use a YAML validator online or in your editor.

### Invalid Configuration Values

**Issue**: `ValidationError: position_size_pct must be between 0 and 1`

**Solution**: Check parameter ranges:
```yaml
risk_management:
  params:
    position_size_pct: 0.02    # 2%, not 2.0 (200%)
    max_exposure_pct: 0.10     # 10%, not 10.0 (1000%)
    stop_loss_pct: 0.02        # 2%, not 2.0 (200%)
```

### Missing Required Fields

**Issue**: `KeyError: 'type' field is required`

**Solution**: Check required fields in configuration:
```yaml
# Every major section needs a 'type' field
workflow:
  type: "backtest"  # Required
  
strategies:
  - type: "momentum"  # Required
  
risk_management:
  type: "fixed"       # Required
```

## 🔄 Runtime Issues

### Container Creation Failures

**Issue**: `ContainerCreationError: Failed to initialize BacktestContainer`

**Solution**: Check logs for specific error:
```bash
# Run with debug logging
python main.py config/simple_backtest.yaml --verbose

# Check for common issues:
# 1. Invalid strategy type
# 2. Missing required parameters
# 3. Conflicting configurations
```

### Memory Issues

**Issue**: `MemoryError` during backtest execution

**Solution**:
```bash
# Check system memory
free -h  # Linux
vm_stat | grep "free\|inactive"  # macOS

# Reduce memory usage in config
workflow:
  infrastructure:
    max_workers: 4        # Reduce from default 8
    memory_limit_gb: 8    # Set explicit limit
    
data:
  chunk_size: 5000        # Smaller chunks
```

### Performance Issues

**Issue**: Backtest runs very slowly

**Solution**:
```yaml
# Enable signal replay for optimization
workflow:
  type: "optimization"
  signal_replay: true     # 10-100x faster
  
# Or reduce data size
data:
  start_date: "2023-06-01"  # 6 months instead of years
  end_date: "2023-12-31"
  
# Optimize strategy parameters
strategies:
  - type: "momentum"
    params:
      calculation_method: "fast"  # Use faster indicators
```

## 🔍 Debugging Tips

### Enable Detailed Logging

```bash
# Run with maximum verbosity
python main.py config/simple_backtest.yaml --verbose --debug

# Or set environment variable
export ADMF_DEBUG=true
python main.py config/simple_backtest.yaml
```

### Check Component Loading

```python
# Test component loading
python -c "
from src.core.components import registry
print(f'Loaded {len(registry.components)} components')
for name, component in registry.components.items():
    print(f'  {name}: {component}')
"
```

### Validate Configuration

```bash
# Test configuration without running
python -c "
import yaml
from src.core.config.schemas import ConfigSchema

with open('config/simple_backtest.yaml') as f:
    config = yaml.safe_load(f)
    
result = ConfigSchema.validate(config)
print('Configuration valid!' if result.is_valid else f'Errors: {result.errors}')
"
```

### Test Individual Components

```python
# Test strategy component
from src.strategy.strategies.momentum import MomentumStrategy

strategy = MomentumStrategy(fast_period=10, slow_period=20)
print(f'Strategy created: {strategy}')
```

## 🌐 Environment-Specific Issues

### Windows Issues

**Issue**: Path separator problems

**Solution**:
```yaml
# Use forward slashes even on Windows
data:
  source:
    path: "data/SPY_1m.csv"  # ✅ Works on all platforms
    # Not: "data\\SPY_1m.csv"  # ❌ Windows only
```

**Issue**: Long path names

**Solution**:
```bash
# Enable long paths on Windows 10+
# Run as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### macOS Issues

**Issue**: "command not found: python"

**Solution**:
```bash
# Use python3 explicitly
python3 main.py config/simple_backtest.yaml

# Or create alias
echo "alias python=python3" >> ~/.zshrc
source ~/.zshrc
```

**Issue**: Apple Silicon (M1/M2) compatibility

**Solution**:
```bash
# Install Rosetta if needed
/usr/sbin/softwareupdate --install-rosetta

# Use conda for better M1 support
conda create -n admf python=3.11
conda activate admf
pip install -r requirements.txt
```

### Linux Issues

**Issue**: Missing system dependencies

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc

# Alpine
apk add python3-dev gcc musl-dev
```

## 📝 Getting More Help

### Log Analysis

When reporting issues, include:
```bash
# System information
python --version
pip list | grep -E "(pandas|numpy|pyyaml)"
uname -a  # Linux/macOS
systeminfo  # Windows

# Error details
python main.py config/simple_backtest.yaml --debug > debug.log 2>&1
# Include debug.log in your report
```

### Creating Minimal Reproduction

Create a minimal config that reproduces the issue:
```yaml
# minimal_test.yaml
workflow:
  type: "backtest"
  
data:
  source:
    type: "csv"
    path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-01-05"  # Just a few days
  
strategies:
  - type: "momentum"
    params:
      fast_period: 5
      slow_period: 10
```

### Common Issue Checklist

Before asking for help, verify:
- [ ] Python 3.11+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Running from project root directory
- [ ] Data files exist in `data/` directory
- [ ] YAML syntax is valid
- [ ] Configuration values are within valid ranges

### Where to Get Help

1. **Documentation**: Check the relevant guides
   - [Core Concepts](../02-core-concepts/README.md) for architecture questions
   - [User Guide](../03-user-guide/README.md) for usage questions
   - [Examples](../09-examples/README.md) for working configurations

2. **GitHub Issues**: For bugs and feature requests
   - Search existing issues first
   - Include minimal reproduction case
   - Provide complete error logs

3. **Community**: For general questions and discussions

---

Still having issues? Check the [Advanced Topics](../08-advanced-topics/README.md) or return to [Getting Started](README.md) ←