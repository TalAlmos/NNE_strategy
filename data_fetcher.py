import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_intraday_data(ticker, date, interval='1m'):
    """
    Fetches intraday data for a specific ticker and date.
    
    Args:
        ticker (str): Stock ticker symbol
        date (str): Date in format 'YYYY-MM-DD'
        interval (str): Data interval (default '1m' for 1 minute)
    
    Returns:
        pandas.DataFrame: DataFrame with the intraday data, or None if error
    """
    try:
        # Convert date string to datetime
        target_date = pd.to_datetime(date)
        
        # Set time range (full trading day)
        start_date = target_date.replace(hour=9, minute=30)
        end_date = target_date.replace(hour=16, minute=0)
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch data
        df = stock.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        # Basic data validation
        if df.empty:
            print(f"No data available for {ticker} on {date}")
            return None
            
        # Reset index to make datetime a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'Datetime'})
        
        # Keep only required columns
        required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]  # This will drop any other columns including Dividends and Stock Splits
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                return None
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_data(df, ticker, date, output_dir='data'):
    """
    Saves the fetched data to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        ticker (str): Stock ticker symbol
        date (str): Date string
        output_dir (str): Directory to save the file (default 'data')
    
    Returns:
        str: Path to saved file, or None if error
    """
    try:
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create full path to stock_raw_data directory
        output_dir = os.path.join(script_dir, 'data', 'stock_raw_data')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create filename using NNE convention
        filename = f"NNE_data_{date.replace('-', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        
        return filepath
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return None

def fetch_and_save_data(ticker, date):
    """
    Main function to fetch and save intraday data.
    
    Args:
        ticker (str): Stock ticker symbol
        date (str): Date in format 'YYYY-MM-DD'
    
    Returns:
        tuple: (DataFrame, filepath) or (None, None) if error
    """
    # Fetch data
    df = fetch_intraday_data(ticker, date)
    if df is None:
        return None, None
    
    # Save data
    filepath = save_data(df, ticker, date)
    if filepath is None:
        return df, None
    
    return df, filepath

def test_data_fetch():
    """
    Test function to verify data fetching for NNE on December 5th, 2024
    """
    # Test parameters
    TICKER = "NNE"
    DATE = "2024-12-11"  # Updated to 2024
    
    print(f"\nTesting data fetch for {TICKER} on {DATE}")
    print("=" * 50)
    
    # Fetch data
    df, filepath = fetch_and_save_data(TICKER, DATE)
    
    if df is not None:
        # Basic data validation
        print("\nData Validation:")
        print(f"Number of rows: {len(df)}")
        print(f"Time range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        print("\nFirst 5 minutes of trading:")
        print(df.head())
        print("\nLast 5 minutes of trading:")
        print(df.tail())
        
        # Check for data gaps
        time_diff = df['Datetime'].diff()
        gaps = time_diff[time_diff > pd.Timedelta(minutes=1)]
        if not gaps.empty:
            print("\nWarning: Found gaps in data:")
            for idx in gaps.index:
                print(f"Gap at {df['Datetime'][idx]}, duration: {gaps[idx]}")
    else:
        print("Error: Failed to fetch data")

if __name__ == "__main__":
    import sys
    
    # Default date is today
    DATE = datetime.now().strftime('%Y-%m-%d')
    
    # If date provided as argument, use it
    if len(sys.argv) > 1:
        try:
            # Convert YYYYMMDD to YYYY-MM-DD
            input_date = sys.argv[1]
            DATE = f"{input_date[:4]}-{input_date[4:6]}-{input_date[6:]}"
        except Exception as e:
            print(f"Error parsing date argument: {e}")
            print("Please use format YYYYMMDD (e.g., 20241211)")
            sys.exit(1)
    
    TICKER = "NNE"
    print(f"\nFetching data for {TICKER} on {DATE}")
    print("=" * 50)
    
    test_data_fetch() 