"""
Helper functions for EOD closing logic that adapt to different timeframes.
"""

def get_eod_bars_for_timeframe(timeframe_minutes: int) -> tuple[int, int]:
    """
    Calculate the bar numbers for EOD cutoffs based on timeframe.
    
    Args:
        timeframe_minutes: The timeframe in minutes (1, 5, 15, 30, 60, etc.)
        
    Returns:
        Tuple of (entry_cutoff_bar, exit_bar) for 3:30 PM and 3:50 PM respectively
    """
    # Market opens at 9:30 AM
    # 3:30 PM is 360 minutes after open
    # 3:50 PM is 380 minutes after open (corrected from 390)
    # 4:00 PM is 390 minutes after open
    
    entry_cutoff_minutes = 360  # 6 hours after 9:30 AM = 3:30 PM
    exit_minutes = 380  # 6 hours 20 min after 9:30 AM = 3:50 PM
    
    # Calculate bar numbers (0-based index)
    entry_cutoff_bar = entry_cutoff_minutes // timeframe_minutes
    exit_bar = exit_minutes // timeframe_minutes
    
    return entry_cutoff_bar, exit_bar


def create_eod_filter_for_timeframe(timeframe_minutes: int, existing_filter: str = None) -> str:
    """
    Create an EOD filter expression for the given timeframe.
    
    Uses time-based filtering which works better with extended hours data.
    
    Args:
        timeframe_minutes: The timeframe in minutes
        existing_filter: Optional existing filter to combine with
        
    Returns:
        Filter expression string
    """
    # Use time-based filter instead of bar_of_day
    # 3:30 PM = 1530, 3:50 PM = 1550
    entry_cutoff_time = 1530  # No new entries after 3:30 PM
    exit_time = 1550  # Force exit at 3:50 PM
    
    # Create the EOD filter using time variable (HHMM format)
    # This forces signal rejection if time >= exit_time
    # The filter context has access to 'signal_value' from the result, not 'signal'
    # We want the filter to return False (reject signal) if time >= 1550
    eod_filter = f"time < {exit_time}"
    
    if existing_filter:
        # Combine with existing filter using AND
        # Both conditions must be true for signal to pass
        return f"({existing_filter}) and ({eod_filter})"
    else:
        # Just the EOD filter
        return eod_filter


# Timeframe detection helper
def detect_timeframe_from_config(config: dict) -> int:
    """
    Detect the timeframe from configuration or data settings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Timeframe in minutes (defaults to 5 if not found)
    """
    # Check various places where timeframe might be specified
    
    # Check data config
    data_config = config.get('data', {})
    if 'timeframe' in data_config:
        return data_config['timeframe']
    
    if 'timeframe_minutes' in data_config:
        return data_config['timeframe_minutes']
    
    # Check for timeframe in filename patterns
    sources = data_config.get('sources', {})
    for source_name, source_config in sources.items():
        if 'path' in source_config:
            path = source_config['path']
            # Extract timeframe from filename
            if '_1m' in path or '_1min' in path:
                return 1
            elif '_5m' in path or '_5min' in path:
                return 5
            elif '_15m' in path or '_15min' in path:
                return 15
            elif '_30m' in path or '_30min' in path:
                return 30
            elif '_60m' in path or '_1h' in path:
                return 60
    
    # Default to 5-minute bars
    return 5