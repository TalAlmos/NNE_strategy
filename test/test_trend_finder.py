from data_fetcher import fetch_and_save_data
from trend_finder import analyze_trends, save_trend_analysis
import pandas as pd

def test_trend_analysis():
    """
    Test function to verify trend analysis with major swing detection
    """
    # Test parameters
    TICKER = "NNE"
    DATE = "2024-12-10"
    
    print(f"\nTesting trend analysis for {TICKER} on {DATE}")
    print("=" * 50)
    
    # Step 1: Fetch Data
    print("\nStep 1: Fetching Data...")
    df, filepath = fetch_and_save_data(TICKER, DATE)
    
    if df is not None:
        # Step 2: Analyze Trends
        print("\nStep 2: Analyzing Trends...")
        df_with_trends = analyze_trends(df)
        
        if df_with_trends is not None:
            # Save detailed analysis
            output_file = f"test_trend_analysis_{TICKER}_{DATE.replace('-', '')}.csv"
            save_trend_analysis(df_with_trends, output_file)
            
            # Print major trend analysis
            print("\nMajor Trend Analysis:")
            trend_changes = df_with_trends['Trend'].ne(df_with_trends['Trend'].shift()).cumsum()
            trend_groups = df_with_trends.groupby(trend_changes)
            
            for trend_num, group in trend_groups:
                if group['Trend'].iloc[0] is None:
                    continue
                
                start_time = group['Datetime'].iloc[0]
                end_time = group['Datetime'].iloc[-1]
                start_price = group['Open'].iloc[0]
                end_price = group['Close'].iloc[-1]
                duration = end_time - start_time
                price_change = end_price - start_price
                price_range = group['High'].max() - group['Low'].min()
                
                # Only show significant trends (duration > 3 mins or price change > $0.20)
                if len(group) > 3 or abs(price_change) >= 0.20:
                    print(f"\nTrend #{trend_num}")
                    print(f"Direction: {group['Trend'].iloc[0]}")
                    print(f"Time: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
                    print(f"Price: ${start_price:.2f} -> ${end_price:.2f}")
                    print(f"Change: ${price_change:.2f}")
                    print(f"Range: ${price_range:.2f}")
                    print(f"Duration: {duration}")
            
            return df_with_trends
    
    print("Error: Failed to complete trend analysis")
    return None

if __name__ == "__main__":
    df_analyzed = test_trend_analysis() 