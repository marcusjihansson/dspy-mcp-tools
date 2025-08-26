import time
import logging
from pathlib import Path 
import json 
from functools import wraps
from typing import Any, Callable, Optional
import hashlib
import pandas as pd
import pickle


# ============================================================================
# Utility Functions
# ============================================================================

def retry_with_backoff(func: Optional[Callable] = None, *, 
                      max_retries: int = 3, 
                      retry_delay: float = 1.0,
                      logger: Optional[logging.Logger] = None):
    """
    Decorator for retrying functions with exponential backoff.
    Can be used with or without parameters.
    
    Usage:
        @retry_with_backoff
        def my_function():
            pass
            
        @retry_with_backoff(max_retries=5, retry_delay=2.0)
        def my_function():
            pass
            
        @retry_with_backoff(logger=my_custom_logger)
        def my_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Use provided logger or try to get from instance
            nonlocal logger
            
            if logger is None and args:
                if hasattr(args[0], 'logger'):
                    logger = args[0].logger
                elif hasattr(args[0], 'config') and hasattr(args[0].config, 'logger'):
                    logger = args[0].config.logger
            
            # If still no logger, use default
            if logger is None:
                logger = logging.getLogger(__name__)
            
            for attempt in range(max_retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1} failed. Retrying in {wait_time:.1f}s. Error: {str(e)}")
                    time.sleep(wait_time)
        return wrapper
    
    # Handle both @retry_with_backoff and @retry_with_backoff() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


def cache_result(func: Optional[Callable] = None, *, 
                cache_dir: str = ".cache", 
                use_cache: bool = True,
                logger: Optional[logging.Logger] = None):
    """
    Decorator for caching function results to disk.
    Supports JSON-serializable objects and pandas DataFrames.
    Can be used with or without parameters.
    
    Usage:
        @cache_result
        def my_function():
            pass
            
        @cache_result(cache_dir="my_cache", use_cache=False)
        def my_function():
            pass
            
        @cache_result(logger=my_custom_logger)
        def my_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Use provided logger or try to get from instance
            nonlocal logger
            config = None
            
            if args and hasattr(args[0], 'config'):
                config = args[0].config
                if logger is None and hasattr(config, 'logger'):
                    logger = config.logger
                # Override decorator params with config values if available
                nonlocal use_cache, cache_dir
                use_cache = getattr(config, 'use_cache', use_cache)
                cache_dir = getattr(config, 'cache_dir', cache_dir)
            elif logger is None and args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            
            # If still no logger, use default
            if logger is None:
                logger = logging.getLogger(__name__)
            
            if not use_cache:
                return f(*args, **kwargs)

            # Create a stable cache key
            args_str = str([str(arg) for arg in args if not hasattr(arg, '__dict__')])
            kwargs_str = str(sorted(kwargs.items()))
            cache_string = f"{f.__module__}.{f.__name__}_{args_str}_{kwargs_str}"
            
            # Use SHA256 for a stable hash
            cache_hash = hashlib.sha256(cache_string.encode()).hexdigest()[:16]
            cache_key = f"{f.__name__}_{cache_hash}"
            
            # Ensure cache directory exists
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Check for existing cache files
            json_cache_file = cache_path / f"{cache_key}.json"
            parquet_cache_file = cache_path / f"{cache_key}.parquet"
            pickle_cache_file = cache_path / f"{cache_key}.pkl"
            
            # Try to load from cache
            if json_cache_file.exists():
                logger.debug(f"Loading from JSON cache: {json_cache_file}")
                try:
                    with open(json_cache_file, 'r') as file:
                        return json.load(file)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"JSON cache read failed: {e}. Regenerating...")
            
            elif parquet_cache_file.exists():
                logger.debug(f"Loading from Parquet cache: {parquet_cache_file}")
                try:
                    return pd.read_parquet(parquet_cache_file)
                except Exception as e:
                    logger.warning(f"Parquet cache read failed: {e}. Regenerating...")
            
            elif pickle_cache_file.exists():
                logger.debug(f"Loading from Pickle cache: {pickle_cache_file}")
                try:
                    with open(pickle_cache_file, 'rb') as file:
                        return pickle.load(file)
                except Exception as e:
                    logger.warning(f"Pickle cache read failed: {e}. Regenerating...")

            # Execute the function
            result = f(*args, **kwargs)
            
            # Save result based on type
            try:
                if isinstance(result, pd.DataFrame):
                    # Save DataFrame as Parquet (efficient and preserves types)
                    result.to_parquet(parquet_cache_file)
                    logger.debug(f"Saved DataFrame to cache: {parquet_cache_file}")
                else:
                    # Try JSON first for simple types
                    try:
                        with open(json_cache_file, 'w') as file:
                            json.dump(result, file, indent=2)
                        logger.debug(f"Saved to JSON cache: {json_cache_file}")
                    except (TypeError, ValueError):
                        # Fall back to pickle for complex objects
                        with open(pickle_cache_file, 'wb') as file:
                            pickle.dump(result, file)
                        logger.debug(f"Saved to Pickle cache: {pickle_cache_file}")
                        
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
            
            return result
        return wrapper
    
    # Handle both @cache_result and @cache_result() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


if __name__ == "__main__":

    from log import LoggerManager
    import yfinance as yf
    import pandas as pd

    # Create your custom logger
    my_logger = LoggerManager()

    # Use it in the decorator
    @cache_result(logger=my_logger)
    @retry_with_backoff(logger=my_logger, max_retries=5)
    def fetch_data(ticker='AAPL', start='2025-01-01', end='2025-02-01'):
        my_logger = LoggerManager()
        data = yf.download(tickers=ticker, start=start, end=end)
        my_logger.info("Donwloading data")
        return data  

    # Or with multiple parameters
    @cache_result(cache_dir=".cache", logger=my_logger)
    @retry_with_backoff(logger=my_logger, max_retries=5)
    def process_data(df):
        my_logger = LoggerManager()
        close = df['Close']
        returns = close.pct_change()
        my_logger.info("Processing data")
        return returns

    # Test the functions
    df = fetch_data()
    print("Price data")
    print(df)

    returns = process_data(df)  
    print('Returns for AAPL')
    print(returns)

   