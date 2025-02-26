"""
Data fetching module using yfinance with configuration support
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional, List, Tuple, Union, Dict
import logging
from nne_strategy.config.config import config

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, save_dir: Optional[Path] = None):
        """Initialize data fetcher with configuration"""
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            # Set the absolute path correctly
            self.save_dir = Path(r"D:\NNE_strategy\nne_strategy\data\raw")
        
        logger.info(f"Data will be saved to: {self.save_dir.absolute()}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load market hours from config
        start_time = config.get('market', 'hours', 'start')
        end_time = config.get('market', 'hours', 'end')
        self.market_start = tuple(map(int, start_time.split(':')))
        self.market_end = tuple(map(int, end_time.split(':')))
        
        # Load other settings
        self.interval = config.get('market', 'data_interval')
        self.required_columns = config.get('data', 'required_columns')
        
    def fetch_date_range(self,
                        ticker: str,
                        start_date: Union[str, datetime],
                        end_date: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for a date range"""
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        # If no end_date provided, use start_date
        end = pd.to_datetime(end_date) if end_date else start
        
        # Generate list of trading days
        dates = pd.date_range(start, end, freq='B')  # B for business days
        dates = [d.strftime('%Y-%m-%d') for d in dates]
        
        logger.info(f"Fetching data for {ticker} from {dates[0]} to {dates[-1]}")
        
        # Fetch data for each date
        results = {}
        for date in dates:
            try:
                data = self.fetch_intraday_data(ticker, date)
                if data is not None:
                    # Save the data
                    saved_path = self._save_data(data, ticker, date)
                    if saved_path:
                        results[date] = data
                        logger.info(f"Successfully fetched and saved data for {date}")
                    else:
                        logger.error(f"Failed to save data for {date}")
            except Exception as e:
                logger.error(f"Error fetching data for {date}: {str(e)}")
                continue
                
        return results

    def fetch_intraday_data(self, 
                           ticker: str,
                           date: str,
                           interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch intraday data for specific date
        
        Args:
            ticker: Stock symbol
            date: Date in YYYY-MM-DD format
            interval: Optional override for data interval
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Convert date and set time range
            target_date = pd.to_datetime(date)
            start_date = target_date.replace(
                hour=self.market_start[0],
                minute=self.market_start[1]
            )
            end_date = target_date.replace(
                hour=self.market_end[0],
                minute=self.market_end[1]
            )
            
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval=interval or self.interval
            )
            
            if df.empty:
                logger.warning(f"No data available for {ticker} on {date}")
                return None
                
            # Process DataFrame
            df = df.reset_index()
            df = df.rename(columns={'index': 'Datetime'})
            
            # Keep only required columns
            df = df[self.required_columns]
            
            # Validate columns
            if not all(col in df.columns for col in self.required_columns):
                logger.error(f"Missing required columns in data")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
            
    def _save_data(self, 
                   data: pd.DataFrame,
                   ticker: str,
                   date: str) -> Optional[Path]:
        """Save data to CSV using specified naming convention"""
        try:
            # Format date to yyyy-mm-dd
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            # Create filename in format: raw_AGEN_yyyy-mm-dd.csv
            filename = f"raw_{ticker}_{date_str}.csv"
            
            # Create ticker-specific directory
            ticker_dir = self.save_dir / ticker
            ticker_dir.mkdir(exist_ok=True)
            
            filepath = ticker_dir / filename
            
            # Save to CSV
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate fetched data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if valid
        """
        if df is None or df.empty:
            return False
            
        # Check time gaps
        time_diff = df['Datetime'].diff()
        gaps = time_diff[time_diff > pd.Timedelta(minutes=1)]
        if not gaps.empty:
            logger.warning("Found gaps in data:")
            for idx in gaps.index:
                logger.warning(f"Gap at {df['Datetime'][idx]}, duration: {gaps[idx]}")
            return False
            
        return True

def main():
    """Main function for command line usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Get ticker from config or command line
        ticker = config.get('trading', 'symbol')
        
        # Parse start date
        if len(sys.argv) < 2:
            logger.error("Please provide at least a start date (YYYYMMDD)")
            logger.error("Usage: python data_fetcher.py START_DATE [END_DATE] [--ticker SYMBOL]")
            sys.exit(1)
            
        start_date = sys.argv[1]
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        
        # Parse end date if provided
        end_date = None
        if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
            end_input = sys.argv[2]
            end_date = f"{end_input[:4]}-{end_input[4:6]}-{end_input[6:]}"
            
        # Check for ticker override
        if '--ticker' in sys.argv:
            ticker_idx = sys.argv.index('--ticker')
            if ticker_idx + 1 < len(sys.argv):
                ticker = sys.argv[ticker_idx + 1]
        
        # Create data directory
        raw_dir = Path(r"D:\NNE_strategy\nne_strategy\data\raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fetcher and get data
        fetcher = DataFetcher(save_dir=raw_dir)
        results = fetcher.fetch_date_range(ticker, start, end_date)
        
        # Print summary
        logger.info(f"\nFetched data for {ticker}:")
        logger.info(f"Total dates processed: {len(results)}")
        logger.info(f"\nFiles saved to: {raw_dir.absolute()}")
        for date, data in results.items():
            filename = f"raw_{ticker}_{date}.csv"
            logger.info(f"{date}: {len(data)} records -> {filename}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error("Usage: python data_fetcher.py START_DATE [END_DATE] [--ticker SYMBOL]")
        logger.error("Dates should be in YYYYMMDD format")
        sys.exit(1)

if __name__ == "__main__":
    main() 