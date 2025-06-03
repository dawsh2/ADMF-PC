# Installation Guide

This guide covers detailed installation instructions for ADMF-PC across different platforms and environments.

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 8GB (16GB recommended for large-scale optimization)
- **Storage**: 1GB for code + space for data
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)

### Python Dependencies
Core dependencies include:
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyyaml >= 6.0
- pydantic >= 2.0.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0 (optional, for ML features)

## Installation Steps

### 1. Basic Installation

#### macOS/Linux
```bash
# Clone repository
git clone <repository-url>
cd ADMF-PC

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src.core; print('Installation successful!')"
```

#### Windows
```powershell
# Clone repository
git clone <repository-url>
cd ADMF-PC

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src.core; print('Installation successful!')"
```

### 2. Development Installation

For contributors or those wanting to modify the framework:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify
pytest tests/
```

### 3. Docker Installation

For isolated environments:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t admf-pc .
docker run -v $(pwd)/config:/app/config admf-pc config/simple_backtest.yaml
```

## Data Setup

### Download Sample Data
```bash
# Download SPY data for examples
python scripts/download_sample_data.py

# Download additional symbols
python scripts/download_sample_data.py --symbols AAPL,MSFT,GOOGL
```

### Using Your Own Data
Place CSV files in the `data/` directory with this format:
```csv
timestamp,open,high,low,close,volume
2023-01-01 09:30:00,400.0,401.5,399.5,401.0,1000000
```

## Environment Configuration

### Virtual Environment Best Practices

Always use a virtual environment to avoid conflicts:

```bash
# Create environment with specific Python version
python3.11 -m venv venv_admf

# Activate and verify version
source venv_admf/bin/activate
python --version  # Should show 3.11.x
```

### Environment Variables

Optional environment variables for advanced usage:

```bash
# Set data directory
export ADMF_DATA_DIR=/path/to/market/data

# Set configuration directory
export ADMF_CONFIG_DIR=/path/to/configs

# Enable debug logging
export ADMF_DEBUG=true
```

## Platform-Specific Notes

### macOS Apple Silicon (M1/M2)
Some dependencies may require Rosetta or specific builds:
```bash
# Install Rosetta if needed
softwareupdate --install-rosetta

# Use platform-specific wheels
pip install --upgrade --force-reinstall numpy pandas
```

### Linux
Ensure system dependencies are installed:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Red Hat/CentOS
sudo yum install python3-devel gcc
```

### Windows
- Use PowerShell or Command Prompt as Administrator for installation
- Ensure Windows Defender exclusions for better performance
- Consider WSL2 for Linux-like environment

## Verification

After installation, verify everything works:

```bash
# Run unit tests
pytest tests/unit/

# Run integration test
python main.py config/test_simple.yaml --bars 10

# Check component loading
python -c "from src.core.components import registry; print(f'Loaded {len(registry.components)} components')"
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Permission Errors
```bash
# Fix permissions on Unix systems
chmod -R 755 scripts/
chmod +x scripts/*.py
```

#### Memory Issues
For large-scale operations, increase Python's memory limit:
```bash
# Linux/macOS
ulimit -v unlimited
```

### Getting Help

If you encounter issues:
1. Check [Troubleshooting Guide](troubleshooting.md)
2. Search existing issues in the repository
3. Create a new issue with:
   - Python version (`python --version`)
   - Platform details
   - Complete error message
   - Steps to reproduce

## Next Steps

✅ Installation complete! Now:
1. [Run your first backtest](first-backtest.md)
2. [Understand the core concepts](../02-core-concepts/README.md)
3. [Explore example configurations](../09-examples/README.md)