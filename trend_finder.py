import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import fetch_and_save_data
import os

def analyze_trends(df):
    """
    Analyzes price movements to identify major trend swings.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC price data
    
    Returns:
        pandas.DataFrame: DataFrame with trend information added
    """
    try:
        # Initialize columns
        df['Trend'] = None
        df['Trend_Start'] = False
        df['Trend_End'] = False
        
        # Parameters for trend identification
        SWING_THRESHOLD = 0.50  # Increased: Minimum price movement to confirm trend change ($)
        CONFIRMATION_PERIODS = 3  # Increased: Number of periods to confirm trend change
        VOLUME_FACTOR = 1.5  # Volume increase factor to help confirm trend change
        MIN_TREND_DURATION = 3  # New: Minimum number of periods for a valid trend
        
        # Calculate moving averages for smoothing
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        
        current_trend = None
        trend_start_idx = 0
        trend_high = float('-inf')
        trend_low = float('inf')
        potential_reversal_idx = None
        avg_volume = df['Volume'].rolling(window=5).mean()
        
        for i in range(len(df)):
            if i < 5:  # Skip until we have enough data for moving averages
                continue
                
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            current_close = df['Close'].iloc[i]
            current_volume = df['Volume'].iloc[i]
            current_sma = df['SMA5'].iloc[i]
            
            if current_trend is None:
                # Initialize first trend
                current_trend = 'UpTrend' if current_close > df['SMA5'].iloc[i-1] else 'DownTrend'
                trend_start_idx = i
                trend_high = current_high
                trend_low = current_low
                df.loc[i, 'Trend_Start'] = True
                df.loc[i, 'Trend'] = current_trend
                continue
            
            # Update trend extremes
            trend_high = max(trend_high, current_high)
            trend_low = min(trend_low, current_low)
            
            # Check for potential trend reversal
            if current_trend == 'UpTrend':
                if current_low < (trend_high - SWING_THRESHOLD):
                    if potential_reversal_idx is None:
                        potential_reversal_idx = i
                    
                    # Confirm reversal with multiple conditions
                    if ((i - potential_reversal_idx) >= CONFIRMATION_PERIODS and 
                        current_volume > avg_volume.iloc[i] * VOLUME_FACTOR and
                        current_sma < df['SMA5'].iloc[i-1]):
                        
                        # Only change trend if the current trend has lasted long enough
                        if (potential_reversal_idx - trend_start_idx) >= MIN_TREND_DURATION:
                            df.loc[trend_start_idx:potential_reversal_idx-1, 'Trend'] = 'UpTrend'
                            df.loc[potential_reversal_idx-1, 'Trend_End'] = True
                            df.loc[potential_reversal_idx, 'Trend_Start'] = True
                            current_trend = 'DownTrend'
                            trend_start_idx = potential_reversal_idx
                            trend_high = current_high
                            trend_low = current_low
                        potential_reversal_idx = None
                else:
                    potential_reversal_idx = None
                    df.loc[i, 'Trend'] = 'UpTrend'
                    
            else:  # DownTrend
                if current_high > (trend_low + SWING_THRESHOLD):
                    if potential_reversal_idx is None:
                        potential_reversal_idx = i
                    
                    # Confirm reversal with multiple conditions
                    if ((i - potential_reversal_idx) >= CONFIRMATION_PERIODS and 
                        current_volume > avg_volume.iloc[i] * VOLUME_FACTOR and
                        current_sma > df['SMA5'].iloc[i-1]):
                        
                        # Only change trend if the current trend has lasted long enough
                        if (potential_reversal_idx - trend_start_idx) >= MIN_TREND_DURATION:
                            df.loc[trend_start_idx:potential_reversal_idx-1, 'Trend'] = 'DownTrend'
                            df.loc[potential_reversal_idx-1, 'Trend_End'] = True
                            df.loc[potential_reversal_idx, 'Trend_Start'] = True
                            current_trend = 'UpTrend'
                            trend_start_idx = potential_reversal_idx
                            trend_high = current_high
                            trend_low = current_low
                        potential_reversal_idx = None
                else:
                    potential_reversal_idx = None
                    df.loc[i, 'Trend'] = 'DownTrend'
        
        # Set trend for last segment
        if trend_start_idx < len(df):
            df.loc[trend_start_idx:, 'Trend'] = current_trend
        
        return df
        
    except Exception as e:
        print(f"Error analyzing trends: {e}")
        return None

def save_trend_analysis(df, date, ticker):
    """
    Saves the trend analysis to CSV and generates a summary.
    
    Args:
        df (pandas.DataFrame): DataFrame with trend analysis
        date (str): Date string in YYYYMMDD format
        ticker (str): Stock ticker symbol
    """
    try:
        # Get script directory and create paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'data', 'stock_trend_complete')
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create file paths
        output_file = os.path.join(output_dir, f"trend_analysis_{ticker}_{date}.csv")
        summary_file = os.path.join(output_dir, f"trend_analysis_{ticker}_{date}_summary.txt")
        
        # Save detailed analysis to CSV
        df.to_csv(output_file, index=False)
        print(f"Trend analysis saved to: {output_file}")
        
        # Generate trend summary
        with open(summary_file, 'w') as f:
            f.write("TREND ANALYSIS SUMMARY\n")
            f.write("=====================\n\n")
            
            # Group by trends
            trend_changes = df['Trend'].ne(df['Trend'].shift()).cumsum()
            for trend_num, group in df.groupby(trend_changes):
                if group['Trend'].iloc[0] == '':
                    continue
                    
                start_time = group['Datetime'].iloc[0]
                end_time = group['Datetime'].iloc[-1]
                start_price = group['Open'].iloc[0]
                end_price = group['Close'].iloc[-1]
                duration = end_time - start_time
                price_change = end_price - start_price
                
                f.write(f"Trend #{trend_num}\n")
                f.write(f"Direction: {group['Trend'].iloc[0]}\n")
                f.write(f"Start: {start_time.strftime('%H:%M')} @ ${start_price:.2f}\n")
                f.write(f"End: {end_time.strftime('%H:%M')} @ ${end_price:.2f}\n")
                f.write(f"Duration: {duration}\n")
                f.write(f"Price Change: ${price_change:.2f}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"Trend summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error saving trend analysis: {e}")

def main(ticker, date):
    """
    Main function to fetch data and perform trend analysis.
    
    Args:
        ticker (str): Stock ticker symbol
        date (str): Date in format 'YYYY-MM-DD'
    """
    # Fetch data using data_fetcher
    df, filepath = fetch_and_save_data(ticker, date)
    
    if df is not None:
        # Perform trend analysis
        df_with_trends = analyze_trends(df)
        
        if df_with_trends is not None:
            # Save analysis with date formatted as YYYYMMDD
            date_formatted = date.replace('-', '')
            save_trend_analysis(df_with_trends, date_formatted, ticker)
            return df_with_trends
    
    return None

if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime
    
    # Get date from command line or use default
    if len(sys.argv) > 1:
        try:
            # Convert YYYYMMDD to YYYY-MM-DD
            input_date = sys.argv[1]
            DATE = f"{input_date[:4]}-{input_date[4:6]}-{input_date[6:]}"
        except Exception as e:
            print(f"Error parsing date argument: {e}")
            print("Please use format YYYYMMDD (e.g., 20241211)")
            sys.exit(1)
    else:
        DATE = "2024-12-11"  # Default date
    
    TICKER = "NNE"
    print(f"\nAnalyzing trends for {TICKER} on {DATE}")
    print("=" * 50)
    
    df_analyzed = main(TICKER, DATE)
    if df_analyzed is not None:
        print("\nAnalysis completed successfully!")
