"""
Data Field Parser

Parses the 'data' field from configs to support clean syntax:
- data: SPY_5m
- data: [SPY_5m, QQQ_5m]
- Fallback to symbols/timeframes fields
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def parse_data_field(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse data field from config and return normalized data specifications.
    
    Supports:
    1. data: "SPY_5m" -> Single file
    2. data: ["SPY_5m", "QQQ_5m"] -> Multiple files
    3. data: {source: "SPY_5m", max_bars: 1000} -> With options
    4. Fallback to symbols/timeframes fields
    
    Returns:
        List of dicts with 'symbol', 'timeframe', and 'file' keys
    """
    data_specs = []
    
    # Check for 'data' field first
    if 'data' in config:
        data_field = config['data']
        
        # Handle string (single file)
        if isinstance(data_field, str):
            data_specs.extend(_parse_data_string(data_field))
            logger.debug(f"Parsed data string '{data_field}' to specs: {data_specs}")
            
        # Handle list (multiple files)
        elif isinstance(data_field, list):
            for item in data_field:
                if isinstance(item, str):
                    data_specs.extend(_parse_data_string(item))
                elif isinstance(item, dict):
                    # Handle dict entries like {source: "SPY_5m", options...}
                    if 'source' in item:
                        specs = _parse_data_string(item['source'])
                        # Add any extra options to each spec
                        for spec in specs:
                            spec.update({k: v for k, v in item.items() if k != 'source'})
                        data_specs.extend(specs)
                        
        # Handle dict with 'source' key
        elif isinstance(data_field, dict) and 'source' in data_field:
            specs = _parse_data_string(data_field['source'])
            # Add any extra options
            for spec in specs:
                spec.update({k: v for k, v in data_field.items() if k != 'source'})
            data_specs.extend(specs)
            
        # Handle dict with 'files' key (alternative syntax)
        elif isinstance(data_field, dict) and 'files' in data_field:
            files = data_field['files']
            if isinstance(files, str):
                data_specs.extend(_parse_data_string(files))
            elif isinstance(files, list):
                for f in files:
                    data_specs.extend(_parse_data_string(f))
                    
    # Fallback to symbols/timeframes fields
    else:
        symbols = config.get('symbols', [])
        timeframes = config.get('timeframes', ['1m'])  # Default to 1m if not specified
        
        # Ensure lists
        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(timeframes, str):
            timeframes = [timeframes]
            
        # Check if symbols already include timeframe (e.g., "SPY_5m")
        for symbol in symbols:
            if '_' in symbol and any(tf in symbol for tf in ['1m', '5m', '15m', '30m', '1h', '1d']):
                # Symbol includes timeframe
                data_specs.extend(_parse_data_string(symbol))
            else:
                # Create combinations of symbols and timeframes
                for timeframe in timeframes:
                    data_specs.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'file': f"{symbol}_{timeframe}"
                    })
    
    # Add common config options to all specs
    common_options = {
        'data_dir': config.get('data_dir', './data'),
        'start_date': config.get('start_date'),
        'end_date': config.get('end_date'),
        'max_bars': config.get('max_bars'),
        'dataset': config.get('dataset'),
        'split_ratio': config.get('split_ratio', 0.8)
    }
    
    for spec in data_specs:
        # Only add options that aren't already in the spec
        for key, value in common_options.items():
            if key not in spec and value is not None:
                spec[key] = value
    
    logger.info(f"Parsed data specifications: {data_specs}")
    return data_specs


def _parse_data_string(data_str: str) -> List[Dict[str, str]]:
    """
    Parse a single data string like "SPY_5m" into components.
    
    Returns:
        List with single dict containing symbol, timeframe, and file
    """
    # Common timeframe patterns
    timeframe_patterns = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', 'daily']
    
    # Try to extract symbol and timeframe
    for tf in timeframe_patterns:
        if data_str.endswith(f'_{tf}'):
            symbol = data_str[:-len(f'_{tf}')]
            return [{
                'symbol': data_str,  # Use full string as symbol for direct file loading
                'timeframe': tf,
                'file': data_str,  # Original string is the file name
                'parsed_symbol': symbol,  # Store parsed symbol separately if needed
                'parsed_timeframe': tf
            }]
    
    # If no timeframe pattern found, treat the whole string as symbol
    # This handles cases like "SPY" or custom naming
    return [{
        'symbol': data_str,
        'timeframe': '1m',  # Default timeframe
        'file': data_str
    }]


def get_data_files(config: Dict[str, Any]) -> List[str]:
    """
    Get list of data files from config.
    
    Returns:
        List of file names (without path or extension)
    """
    data_specs = parse_data_field(config)
    return [spec['file'] for spec in data_specs]


def get_symbols_from_data(config: Dict[str, Any]) -> List[str]:
    """
    Extract unique symbols from data configuration.
    
    Returns:
        List of unique symbols
    """
    data_specs = parse_data_field(config)
    symbols = list(set(spec['symbol'] for spec in data_specs))
    return symbols