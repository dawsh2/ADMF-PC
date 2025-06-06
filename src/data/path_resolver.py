"""
Data Path Resolver for automatic path inference based on symbol and timeframe.

This module provides intelligent path resolution for market data files,
supporting various naming conventions and timeframe formats.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataPathResolver:
    """
    Resolves data file paths based on symbol and timeframe.
    
    Supports patterns like:
    - SPY_1m.csv for 1 minute data
    - SPY_1h.csv for hourly data
    - SPY_1d.csv for daily data
    - SPY_1M.csv for monthly data
    - SPY.csv as fallback for daily data
    """
    
    # Standard timeframe mappings
    TIMEFRAME_MAPPINGS = {
        # Intraday
        '1m': ['1m', '1min', '1minute', 'minute'],
        '5m': ['5m', '5min', '5minute', '5minutes'],
        '15m': ['15m', '15min', '15minute', '15minutes'],
        '30m': ['30m', '30min', '30minute', '30minutes'],
        '1h': ['1h', '1hr', '1hour', 'hourly', '60m', '60min'],
        '4h': ['4h', '4hr', '4hour', '240m'],
        
        # Daily and above
        '1d': ['1d', '1day', 'daily', 'd', 'day'],
        '1w': ['1w', '1wk', '1week', 'weekly', 'w', 'week'],
        '1M': ['1M', '1mo', '1month', 'monthly', 'M', 'month'],
        '1y': ['1y', '1yr', '1year', 'yearly', 'y', 'year'],
    }
    
    # Reverse mapping for normalization
    TIMEFRAME_REVERSE = {}
    for canonical, variants in TIMEFRAME_MAPPINGS.items():
        for variant in variants:
            TIMEFRAME_REVERSE[variant] = canonical
    
    def __init__(self, 
                 base_dir: str = "data",
                 search_subdirs: bool = True,
                 case_sensitive: bool = False):
        """
        Initialize path resolver.
        
        Args:
            base_dir: Base directory for data files
            search_subdirs: Whether to search subdirectories
            case_sensitive: Whether symbol matching is case sensitive
        """
        self.base_dir = Path(base_dir)
        self.search_subdirs = search_subdirs
        self.case_sensitive = case_sensitive
        
        # Cache for resolved paths
        self._path_cache: Dict[str, Path] = {}
        
        # Build file index on initialization
        self._file_index = self._build_file_index()
        
    def _build_file_index(self) -> Dict[str, List[Path]]:
        """Build an index of available data files."""
        index = {}
        
        if not self.base_dir.exists():
            logger.warning(f"Data directory does not exist: {self.base_dir}")
            return index
        
        # Get all CSV files
        if self.search_subdirs:
            csv_files = list(self.base_dir.rglob("*.csv"))
        else:
            csv_files = list(self.base_dir.glob("*.csv"))
        
        # Index by symbol (extracted from filename)
        for file_path in csv_files:
            filename = file_path.stem  # Without .csv extension
            
            # Extract symbol from filename
            # Handles: SPY_1m, SPY_daily, SPY, etc.
            parts = filename.split('_')
            symbol = parts[0]
            
            if not self.case_sensitive:
                symbol = symbol.upper()
            
            if symbol not in index:
                index[symbol] = []
            index[symbol].append(file_path)
        
        logger.info(f"Indexed {len(csv_files)} data files for {len(index)} symbols")
        return index
    
    def resolve_path(self, symbol: str, timeframe: str) -> Optional[Path]:
        """
        Resolve data file path for given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            
        Returns:
            Path to data file if found, None otherwise
        """
        # Check cache first
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        # Normalize inputs
        if not self.case_sensitive:
            symbol = symbol.upper()
        
        # Normalize timeframe
        normalized_tf = self._normalize_timeframe(timeframe)
        
        # Get candidate files for this symbol
        if symbol not in self._file_index:
            logger.debug(f"No files found for symbol: {symbol}")
            return None
        
        candidate_files = self._file_index[symbol]
        
        # Try exact pattern matches first
        patterns = self._generate_filename_patterns(symbol, normalized_tf, timeframe)
        
        for pattern in patterns:
            for file_path in candidate_files:
                if file_path.stem.lower() == pattern.lower():
                    logger.debug(f"Resolved {symbol} {timeframe} -> {file_path}")
                    self._path_cache[cache_key] = file_path
                    return file_path
        
        # Fallback: Check if we have a simple symbol.csv file
        # This often contains daily data
        if normalized_tf == '1d':
            for file_path in candidate_files:
                if file_path.stem.lower() == symbol.lower():
                    logger.debug(f"Resolved {symbol} {timeframe} -> {file_path} (fallback)")
                    self._path_cache[cache_key] = file_path
                    return file_path
        
        logger.debug(f"Could not resolve path for {symbol} {timeframe}")
        return None
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe to canonical form."""
        # Check exact match first (for case-sensitive timeframes like 1M vs 1m)
        if timeframe in self.TIMEFRAME_REVERSE:
            return self.TIMEFRAME_REVERSE[timeframe]
        
        # Then try lowercase
        tf_lower = timeframe.lower()
        return self.TIMEFRAME_REVERSE.get(tf_lower, timeframe)
    
    def _generate_filename_patterns(self, symbol: str, normalized_tf: str, original_tf: str) -> List[str]:
        """Generate possible filename patterns to search for."""
        patterns = []
        
        # Primary pattern: symbol_timeframe
        patterns.append(f"{symbol}_{normalized_tf}")
        patterns.append(f"{symbol}_{original_tf}")
        
        # Also try all known variants
        if normalized_tf in self.TIMEFRAME_MAPPINGS:
            for variant in self.TIMEFRAME_MAPPINGS[normalized_tf]:
                patterns.append(f"{symbol}_{variant}")
        
        # Special cases
        if normalized_tf == '1d':
            patterns.extend([
                f"{symbol}_daily",
                f"{symbol}_d",
                f"{symbol}_day"
            ])
        elif normalized_tf == '1m':
            patterns.extend([
                f"{symbol}_minute",
                f"{symbol}_1minute"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)
        
        return unique_patterns
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """
        List all available symbol-timeframe combinations.
        
        Returns:
            Dict mapping symbols to list of available timeframes
        """
        available = {}
        
        for symbol, files in self._file_index.items():
            timeframes = []
            
            for file_path in files:
                filename = file_path.stem
                
                # Extract timeframe from filename if present
                if '_' in filename:
                    parts = filename.split('_', 1)
                    if len(parts) > 1:
                        tf_part = parts[1]
                        # Try to normalize it
                        normalized = self._normalize_timeframe(tf_part)
                        if normalized in self.TIMEFRAME_MAPPINGS:
                            timeframes.append(normalized)
                        else:
                            # Might be a descriptor like "daily"
                            for canonical, variants in self.TIMEFRAME_MAPPINGS.items():
                                if tf_part in variants:
                                    timeframes.append(canonical)
                                    break
                else:
                    # Plain symbol.csv - assume daily
                    timeframes.append('1d')
            
            if timeframes:
                available[symbol] = list(set(timeframes))  # Remove duplicates
        
        return available
    
    def suggest_alternatives(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        Suggest alternative data files when exact match not found.
        
        Returns:
            List of alternative suggestions with metadata
        """
        suggestions = []
        
        # Check if we have the symbol at all
        if not self.case_sensitive:
            symbol = symbol.upper()
        
        if symbol in self._file_index:
            # Symbol exists, different timeframe
            available_tfs = []
            for file_path in self._file_index[symbol]:
                filename = file_path.stem
                if '_' in filename:
                    parts = filename.split('_', 1)
                    if len(parts) > 1:
                        tf_part = parts[1]
                        normalized = self._normalize_timeframe(tf_part)
                        if normalized:
                            available_tfs.append({
                                'timeframe': normalized,
                                'path': file_path,
                                'exact_match': False
                            })
                else:
                    # Plain file - assume daily
                    available_tfs.append({
                        'timeframe': '1d',
                        'path': file_path,
                        'exact_match': False
                    })
            
            suggestions.extend(available_tfs)
        
        else:
            # Symbol not found - suggest similar symbols
            all_symbols = list(self._file_index.keys())
            
            # Simple similarity check (starts with same letter, etc.)
            similar = [s for s in all_symbols if s[0] == symbol[0]][:5]
            
            for sim_symbol in similar:
                suggestions.append({
                    'symbol': sim_symbol,
                    'available_files': len(self._file_index[sim_symbol]),
                    'exact_match': False
                })
        
        return suggestions
    
    def refresh_index(self):
        """Refresh the file index (useful if files are added during runtime)."""
        self._file_index = self._build_file_index()
        self._path_cache.clear()
        logger.info("Refreshed data file index")


# Convenience function for simple usage
def resolve_data_path(symbol: str, timeframe: str, base_dir: str = "data") -> Optional[Path]:
    """
    Simple function to resolve data path.
    
    Example:
        path = resolve_data_path("SPY", "1m")  # Returns Path to SPY_1m.csv
        path = resolve_data_path("QQQ", "1d")  # Returns Path to QQQ_1d.csv or QQQ.csv
    """
    resolver = DataPathResolver(base_dir)
    return resolver.resolve_path(symbol, timeframe)