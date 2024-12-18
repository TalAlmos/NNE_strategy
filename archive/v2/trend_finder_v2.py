from datetime import datetime
import pandas as pd
import numpy as np
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
        first_30min = data[
            (data['Datetime'].dt.strftime('%H:%M') >= '09:30') & 
            (data['Datetime'].dt.strftime('%H:%M') < '10:00')
        ]
        
        if first_30min.empty:
            return None
        
        # Calculate average price
        avg_price = first_30min['Close'].mean()
        high_price = first_30min['High'].max()
        low_price = first_30min['Low'].min()
        
        # Find whether high or low is closer to average
        distance_to_high = abs(high_price - avg_price)
        distance_to_low = abs(low_price - avg_price)
        
        # Get the time and price for the trend point
        if distance_to_high < distance_to_low:
            trend_price = high_price
            # Find the time when this high occurred
            trend_time = first_30min.loc[first_30min['High'] == high_price, 'Datetime'].iloc[0]
        else:
            trend_price = low_price
            # Find the time when this low occurred
            trend_time = first_30min.loc[first_30min['Low'] == low_price, 'Datetime'].iloc[0]
        
        # Store and return the trend point
        trend_point = (trend_time, trend_price)
        self.trend_points.append(trend_point)
        
        return trend_point
    
    def find_hourly_trend_point(self, data, hour):
        """Find trend point for given hour"""
        # Filter hour's data
        hour_data = data[
            (data['Datetime'].dt.strftime('%H') == f'{hour:02d}')
        ]
        
        if hour_data.empty:
            return None
        
        # Calculate average price for the hour
        avg_price = hour_data['Close'].mean()
        high_price = hour_data['High'].max()
        low_price = hour_data['Low'].min()
        
        # Find whether high or low is closer to average
        distance_to_high = abs(high_price - avg_price)
        distance_to_low = abs(low_price - avg_price)
        
        # Get the time and price for the trend point
        if distance_to_high < distance_to_low:
            trend_price = high_price
            trend_time = hour_data.loc[hour_data['High'] == high_price, 'Datetime'].iloc[0]
        else:
            trend_price = low_price
            trend_time = hour_data.loc[hour_data['Low'] == low_price, 'Datetime'].iloc[0]
        
        # Store and return the trend point
        trend_point = (trend_time, trend_price)
        self.trend_points.append(trend_point)
        
        return trend_point
    
    def collect_trend_points(self, data):
        """Collect all trend points for the day"""
        # Clear any existing points
        self.trend_points = []
        
        # Get first trend point (9:30-10:00)
        first_point = self.find_first_trend_point(data)
        if not first_point:
            print("Failed to find first trend point")
            return False
        
        print(f"\nFirst trend point: Time={first_point[0].strftime('%H:%M')}, Price=${first_point[1]:.2f}")
        
        # Loop through hours until we have 7 points
        current_hour = 10
        while len(self.trend_points) < self.MIN_POINTS and current_hour < 16:
            point = self.find_hourly_trend_point(data, current_hour)
            if point:
                print(f"Hour {current_hour} trend point: Time={point[0].strftime('%H:%M')}, Price=${point[1]:.2f}")
                current_hour += 1
            else:
                print(f"No trend point found for hour {current_hour}")
                current_hour += 1
                continue
        
        # Check if we collected enough points
        if len(self.trend_points) < self.MIN_POINTS:
            print(f"Warning: Only found {len(self.trend_points)} trend points")
            return False
        
        # Sort points by time to ensure chronological order
        self.trend_points.sort(key=lambda x: x[0])
        
        print(f"\nCollected {len(self.trend_points)} trend points:")
        for time, price in self.trend_points:
            print(f"Time: {time.strftime('%H:%M')}, Price: ${price:.2f}")
        
        return True
    
    def analyze_trends(self):
        """Create and classify trends between points"""
        # Clear existing trends
        self.trends = []
        
        # Need at least 2 points to create trends
        if len(self.trend_points) < 2:
            print("Not enough points to analyze trends")
            return False
        
        # Connect consecutive points and classify trends
        for i in range(len(self.trend_points) - 1):
            start_time, start_price = self.trend_points[i]
            end_time, end_price = self.trend_points[i + 1]
            
            # Determine trend direction
            if end_price > start_price:
                direction = 'UpTrend'
            else:
                direction = 'DownTrend'
            
            # Calculate trend characteristics
            price_change = end_price - start_price
            duration = (end_time - start_time).total_seconds() / 60  # in minutes
            
            # Create trend info
            trend = {
                'start_time': start_time,
                'end_time': end_time,
                'start_price': start_price,
                'end_price': end_price,
                'direction': direction,
                'price_change': price_change,
                'duration': duration
            }
            
            self.trends.append(trend)
            
            # Print trend information
            print(f"\nTrend #{i+1}:")
            print(f"Time: {start_time.strftime('%H:%M')} -> {end_time.strftime('%H:%M')}")
            print(f"Price: ${start_price:.2f} -> ${end_price:.2f}")
            print(f"Direction: {direction}")
            print(f"Change: ${price_change:.2f}")
            print(f"Duration: {duration:.0f} minutes")
        
        return True
    
    def detect_trend(self, data):
        """Main method to find and analyze trends"""
        # Reset any existing analysis
        self.trend_points = []
        self.trends = []
        
        print("\nStarting trend detection...")
        
        # Step 1: Collect trend points
        if not self.collect_trend_points(data):
            print("Failed to collect enough trend points")
            return None
        
        # Step 2: Analyze trends between points
        if not self.analyze_trends():
            print("Failed to analyze trends")
            return None
        
        # Step 3: Create summary of analysis
        summary = {
            'total_points': len(self.trend_points),
            'total_trends': len(self.trends),
            'trends': self.trends,
            'first_point': {
                'time': self.trend_points[0][0],
                'price': self.trend_points[0][1]
            },
            'last_point': {
                'time': self.trend_points[-1][0],
                'price': self.trend_points[-1][1]
            },
            'total_change': self.trend_points[-1][1] - self.trend_points[0][1],
            'trend_sequence': [trend['direction'] for trend in self.trends]
        }
        
        # Print summary
        print("\nTrend Analysis Summary:")
        print("-" * 40)
        print(f"Total Points: {summary['total_points']}")
        print(f"Total Trends: {summary['total_trends']}")
        print(f"First Point: {summary['first_point']['time'].strftime('%H:%M')} @ ${summary['first_point']['price']:.2f}")
        print(f"Last Point: {summary['last_point']['time'].strftime('%H:%M')} @ ${summary['last_point']['price']:.2f}")
        print(f"Total Change: ${summary['total_change']:.2f}")
        print("\nTrend Sequence:")
        for i, direction in enumerate(summary['trend_sequence'], 1):
            print(f"Trend {i}: {direction}")
        
        return summary

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
    
    # Initialize and run trend finder
    finder = TrendFinder()
    summary = finder.detect_trend(data)
    
    if summary:
        # Save results to CSV
        trends_df = pd.DataFrame(finder.trends)
        trends_df.to_csv(output_dir / f"trend_analysis_v2_NNE_{date}.csv", index=False)
        
        # Save summary to TXT
        with open(output_dir / f"trend_analysis_v2_NNE_{date}_summary.txt", 'w') as f:
            f.write(f"NNE Trend Analysis Summary (v2) for {date}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Points: {summary['total_points']}\n")
            f.write(f"Total Trends: {summary['total_trends']}\n")
            f.write(f"First Point: {summary['first_point']['time'].strftime('%H:%M')} @ ${summary['first_point']['price']:.2f}\n")
            f.write(f"Last Point: {summary['last_point']['time'].strftime('%H:%M')} @ ${summary['last_point']['price']:.2f}\n")
            f.write(f"Total Change: ${summary['total_change']:.2f}\n\n")
            
            f.write("Trend Sequence:\n")
            f.write("-" * 30 + "\n")
            for i, trend in enumerate(finder.trends, 1):
                f.write(f"\nTrend #{i}:\n")
                f.write(f"Time: {trend['start_time'].strftime('%H:%M')} -> {trend['end_time'].strftime('%H:%M')}\n")
                f.write(f"Price: ${trend['start_price']:.2f} -> ${trend['end_price']:.2f}\n")
                f.write(f"Direction: {trend['direction']}\n")
                f.write(f"Change: ${trend['price_change']:.2f}\n")
                f.write(f"Duration: {trend['duration']:.0f} minutes\n")
        
        print(f"\nResults saved to data/stock_trend_complete/trend_analysis_v2_NNE_{date}.csv")
        print(f"Summary saved to data/stock_trend_complete/trend_analysis_v2_NNE_{date}_summary.txt")
    else:
        print("\nFailed to complete trend analysis") 