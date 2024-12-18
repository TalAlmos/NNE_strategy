from datetime import datetime
import pandas as pd
import numpy as np
from counter_move_stats import CounterMoveStats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class TrendFinder:
    def __init__(self):
        """Initialize trend finder with new trend point approach"""
        # Basic parameters
        self.trend_points = []  # Will store (time, price) tuples
        self.trends = []        # Will store (start_point, end_point, direction) tuples
        
        # Analysis parameters
        self.MIN_POINTS = 7     # Number of trend points needed
        self.HOURS_TO_CHECK = range(10, 17)  # 10:00 to 16:00
    
    def find_first_trend_point(self, data):
        """Find first trend point from 9:30-10:00 data"""
        # Filter first 30 min data
        # Calculate average
        # Find whether high or low is closer to average
        # Return the trend point
        pass
    
    def find_hourly_trend_point(self, data, hour):
        """Find trend point for given hour"""
        # Filter hour's data
        # Calculate hour's average
        # Find price point closest to average
        # Return the trend point
        pass
    
    def collect_trend_points(self, data):
        """Collect all trend points for the day"""
        # Get first trend point (9:30-10:00)
        # Loop through hours until we have 7 points
        # Store points in self.trend_points
        pass
    
    def analyze_trends(self):
        """Create and classify trends between points"""
        # Connect consecutive points
        # Classify each connection as Up/Down
        # Store in self.trends
        pass
    
    def detect_trend(self, data):
        """Main method to find and analyze trends"""
        # Collect 7 trend points
        # Analyze trends between points
        # Return trend analysis
        pass

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import pandas as pd
    
    # Get date from command line or use default
    date = sys.argv[1] if len(sys.argv) > 1 else "20241212"
    
    # Setup paths
    data_path = Path(__file__).parent / "data" / "stock_raw_data" / f"NNE_data_{date}.csv"
    output_dir = Path(__file__).parent / "data" / "stock_trend_complete"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    
    print(f"\nAnalyzing trends for NNE on {date}")
    print("=" * 50)
    
    # Initialize trend finder
    finder = TrendFinder()
    trends = []
    
    # Initialize trend with first window
    initial_window = data.iloc[:finder.CURVE_FIT_WINDOW].copy()
    if not finder.initialize_trend(initial_window):
        print("Failed to initialize trend")
        sys.exit(1)
    
    # Create plot
    plt.figure(figsize=(15, 8))
    plt.plot(data['Close'], label='Price', color='blue', alpha=0.5)
    
    # Process each bar and plot trends
    for i in range(finder.CURVE_FIT_WINDOW, len(data)):
        window = data.iloc[i-finder.CURVE_FIT_WINDOW:i+1].copy()
        trend_info = finder.detect_trend(window)
        
        if trend_info and trend_info.get('trend_changed', False):
            # Update trend tracking
            finder.current_trend = trend_info['trend']
            finder.trend_start_price = window['Close'].iloc[-1]
            
            # Plot and record trend change
            plt.scatter(i, window['Close'].iloc[-1], 
                       color='red' if trend_info['trend'] == 'DownTrend' else 'green',
                       s=100)
            
            # Plot fitted curve
            x_curve = range(i-finder.CURVE_FIT_WINDOW, i+1)
            plt.plot(x_curve, trend_info['curve_values'], '--', alpha=0.5)
            
            trends.append({
                'time': window.iloc[-1]['Datetime'],
                'price': window.iloc[-1]['Close'],
                'trend': trend_info['trend'],
                'strength': trend_info['strength'],
                'curvature': trend_info['curvature'],
                'volume_ratio': trend_info['volume_ratio'],
                'counter_move': trend_info['counter_move']
            })
    
    plt.title(f'NNE Price and Trend Changes - {date}')
    plt.grid(True)
    plt.savefig(output_dir / f'trend_analysis_NNE_{date}.png')
    plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(trends)
    if not results_df.empty:
        results_df['time'] = pd.to_datetime(results_df['time'])
        results_df['datetime_str'] = results_df['time'].dt.strftime('%H:%M')
        results_df = results_df[[
            'datetime_str', 
            'trend', 
            'price', 
            'strength',
            'curvature',
            'volume_ratio',
            'counter_move'
        ]]
        results_df.columns = [
            'Time',
            'Trend',
            'Price',
            'Trend Strength',
            'Curvature',
            'Volume Ratio',
            'Counter Move'
        ]
        results_df.to_csv(output_dir / f"trend_analysis_NNE_{date}.csv", index=False, float_format='%.4f')
    else:
        # Create empty CSV with headers
        pd.DataFrame(columns=[
            'Time',
            'Trend',
            'Price',
            'Trend Strength',
            'Curvature',
            'Volume Ratio',
            'Counter Move'
        ]).to_csv(output_dir / f"trend_analysis_NNE_{date}.csv", index=False)

    # Save detailed summary to TXT
    with open(output_dir / f"trend_analysis_NNE_{date}_summary.txt", 'w') as f:
        f.write(f"NNE Trend Analysis Summary for {date}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total trends detected: {len(trends)}\n")
        f.write(f"Initial trend: {finder.current_trend}\n")
        f.write(f"Initial price: ${finder.trend_start_price:.2f}\n\n")
        
        if trends:
            f.write("Trend Changes:\n")
            f.write("-" * 30 + "\n")
            for i, trend in enumerate(trends, 1):
                f.write(f"\nTrend Change #{i}\n")
                f.write(f"Time: {pd.to_datetime(trend['time']).strftime('%H:%M')}\n")
                f.write(f"Direction: {trend['trend']}\n")
                f.write(f"Price: ${trend['price']:.2f}\n")
                f.write(f"Strength: {trend['strength']:.4f}\n")
                f.write(f"Curvature: {trend['curvature']:.4f}\n")
                f.write(f"Volume Ratio: {trend['volume_ratio']:.2f}\n")
                f.write(f"Counter Move: ${abs(trend['counter_move']):.2f}\n")
                f.write("-" * 30 + "\n")
            
            # Add summary statistics
            f.write("\nSummary Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average trend duration: {(trends[-1]['time'] - trends[0]['time']).total_seconds()/60/len(trends):.1f} minutes\n")
            f.write(f"Total price movement: ${abs(trends[-1]['price'] - finder.trend_start_price):.2f}\n")
            f.write(f"Average counter move: ${results_df['Counter Move'].mean():.2f}\n")
        else:
            f.write("\nNo trend changes detected.\n")
            f.write("Consider adjusting detection parameters:\n")
            f.write(f"- Current TREND_THRESHOLD: {finder.TREND_THRESHOLD}\n")
            f.write(f"- Current CURVATURE_THRESHOLD: {finder.CURVATURE_THRESHOLD}\n")
            f.write(f"- Current VOLUME_THRESHOLD: {finder.VOLUME_THRESHOLD}\n")
    
    print(f"\nAnalysis complete. Found {len(trends)} trend changes.")
    print(f"Results saved to data/stock_trend_complete/trend_analysis_NNE_{date}.csv")
    print(f"Summary saved to data/stock_trend_complete/trend_analysis_NNE_{date}_summary.txt")
