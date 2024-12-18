import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TrendAnalysis:
    def __init__(self):
        """Initialize analysis parameters"""
        # Core parameters
        self.min_trend_duration = 3  # minutes
        self.min_price_change = 0.15  # percent
        self.min_volume = 500
        
    def analyze_data(self, data_path: str, symbol: str = 'NNE'):
        """
        Analyze price data for trends
        
        Args:
            data_path: Path to raw data file
            symbol: Stock symbol
        """
        print(f"\nAnalyzing data from: {data_path}")
        
        # Load and prepare data
        data = self._load_data(data_path)
        if data is None:
            print("Failed to load data")
            return None
        
        print(f"Loaded {len(data)} data points")
        print("Data columns:", data.columns.tolist())
        print("Index type:", type(data.index))
        print("First few rows:")
        print(data.head())
        
        try:
            # Calculate basic metrics
            data = self._calculate_metrics(data)
            print("\nCalculated metrics:")
            print(data[['Price_Change', 'Pct_Change', 'Volume_MA']].head())
            
            # Identify major moves
            major_moves = self._identify_major_moves(data)
            print(f"\nIdentified {len(major_moves)} major price moves")
            if major_moves:
                print("First major move:", major_moves[0])
            
            # Detect trends
            trends = self._detect_trends(data, major_moves)
            print(f"Detected {len(trends)} trends")
            if trends:
                print("First trend:", trends[0])
            
            # Generate analysis
            self._generate_report(data, trends, symbol)
            self._generate_plots(data, trends, symbol)
            
            # Save trend-annotated data
            self._save_trend_data(data, trends, symbol)
            
            return {
                'trends': trends,
                'statistics': self._calculate_statistics(data, trends)
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate price data"""
        try:
            data = pd.read_csv(data_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def _calculate_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price changes and basic metrics"""
        data['Price_Change'] = data['Close'].diff()
        data['Pct_Change'] = data['Close'].pct_change() * 100
        data['Volume_MA'] = data['Volume'].rolling(5).mean()
        return data
        
    def _identify_major_moves(self, data: pd.DataFrame) -> list:
        """Identify significant price movements"""
        major_moves = []
        
        for i in range(1, len(data)):
            if abs(data['Pct_Change'].iloc[i]) > self.min_price_change:
                move = {
                    'time': data.index[i],
                    'price': data['Close'].iloc[i],
                    'change': data['Price_Change'].iloc[i],
                    'pct_change': data['Pct_Change'].iloc[i],
                    'volume': data['Volume'].iloc[i]
                }
                major_moves.append(move)
                
        return major_moves
        
    def _detect_trends(self, data: pd.DataFrame, major_moves: list) -> list:
        """Detect and classify major price trends"""
        trends = []
        current_trend = None
        trend_start_idx = 0
        trend_high = float('-inf')
        trend_low = float('inf')
        reversal_threshold = 0.40  # Minimum price change to consider a trend reversal
        continuation_threshold = 0.15  # Minimum move to continue trend
        
        for i in range(1, len(data)):
            price = data['Close'].iloc[i]
            price_change = data['Price_Change'].iloc[i]
            volume = data['Volume'].iloc[i]
            
            if current_trend is None:
                # Start new trend on significant move
                if abs(price_change) >= continuation_threshold and volume >= self.min_volume:
                    current_trend = 'Up' if price_change > 0 else 'Down'
                    trend_start_idx = i-1  # Include the bar before the move
                    trend_high = max(data['High'].iloc[trend_start_idx:i+1])
                    trend_low = min(data['Low'].iloc[trend_start_idx:i+1])
                    continue
            
            else:
                # Update trend extremes
                trend_high = max(trend_high, data['High'].iloc[i])
                trend_low = min(trend_low, data['Low'].iloc[i])
                
                # Check for trend continuation or reversal
                if current_trend == 'Up':
                    # Price made new high or holding above trend low
                    if price >= trend_high - continuation_threshold:
                        continue
                    # Significant reversal
                    elif trend_high - price >= reversal_threshold:
                        # Record completed uptrend
                        trend_data = data.iloc[trend_start_idx:i+1]
                        trends.append(self._create_trend_record('Up', trend_data))
                        # Start new downtrend
                        current_trend = 'Down'
                        trend_start_idx = i
                        trend_high = price
                        trend_low = price
                
                else:  # current_trend == 'Down'
                    # Price made new low or holding below trend high
                    if price <= trend_low + continuation_threshold:
                        continue
                    # Significant reversal
                    elif price - trend_low >= reversal_threshold:
                        # Record completed downtrend
                        trend_data = data.iloc[trend_start_idx:i+1]
                        trends.append(self._create_trend_record('Down', trend_data))
                        # Start new uptrend
                        current_trend = 'Up'
                        trend_start_idx = i
                        trend_high = price
                        trend_low = price
        
        # Add final trend if exists
        if current_trend:
            trend_data = data.iloc[trend_start_idx:]
            if len(trend_data) >= 3:  # Minimum trend duration
                trends.append(self._create_trend_record(current_trend, trend_data))
        
        # Sort chronologically
        sorted_trends = sorted(trends, key=lambda x: x['start_time'])
        
        # Print summary table
        self._print_trend_summary(sorted_trends)
        
        return sorted_trends
        
    def _create_trend_record(self, direction: str, trend_data: pd.DataFrame) -> dict:
        """Create a trend record from trend data"""
        return {
            'direction': direction,
            'start_time': trend_data.index[0],
            'end_time': trend_data.index[-1],
            'start_price': trend_data['Close'].iloc[0],
            'end_price': trend_data['Close'].iloc[-1],
            'high': trend_data['High'].max(),
            'low': trend_data['Low'].min(),
            'duration': len(trend_data),
            'price_change': trend_data['Close'].iloc[-1] - trend_data['Close'].iloc[0],
            'pct_change': (trend_data['Close'].iloc[-1] / trend_data['Close'].iloc[0] - 1) * 100,
            'avg_volume': trend_data['Volume'].mean(),
            'total_volume': trend_data['Volume'].sum(),
            'volatility': trend_data['Close'].std(),
            'momentum': (trend_data['Close'].iloc[-1] - trend_data['Close'].iloc[0]) / len(trend_data),
            'volume_momentum': trend_data['Volume'].mean() / trend_data['Volume'].iloc[0]
        }
        
    def _print_trend_summary(self, trends: list):
        """Print trend summary table"""
        print("\nTrend Summary (Chronological Order):")
        print("-" * 100)
        print("Dir    | Time          | Price Range      | Dur | Change    | Vol      | Mom     | V.Mom  ")
        print("-" * 100)
        
        for trend in trends:
            print("{:<6} | {:>8}-{:<8} | ${:>7.2f}-${:<7.2f} | {:>3}m | ${:>6.2f} | {:>8.0f} | {:>7.3f} | {:>6.1f}"
                  .format(
                      trend['direction'],
                      trend['start_time'].strftime('%H:%M'),
                      trend['end_time'].strftime('%H:%M'),
                      trend['start_price'],
                      trend['end_price'],
                      trend['duration'],
                      trend['price_change'],
                      trend['avg_volume'],
                      trend['momentum'],
                      trend['volume_momentum']
                  ))

    def _generate_report(self, data: pd.DataFrame, trends: list, symbol: str):
        """Generate analysis report"""
        # Create report directory if it doesn't exist
        report_dir = Path('data/trend_analysis')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report filename with date
        date_str = data.index[0].strftime('%Y%m%d')
        report_path = report_dir / f"trend_analysis_{symbol}_{date_str}.txt"
        
        with open(report_path, 'w') as f:
            # Write header
            f.write(f"Trend Analysis Report for {symbol}\n")
            f.write(f"Date: {date_str}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write summary statistics
            f.write("Summary Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Trends: {len(trends)}\n")
            f.write(f"Trading Period: {data.index[0].strftime('%H:%M')} - {data.index[-1].strftime('%H:%M')}\n")
            f.write(f"Opening Price: ${data['Open'].iloc[0]:.2f}\n")
            f.write(f"Closing Price: ${data['Close'].iloc[-1]:.2f}\n")
            f.write(f"Day's Range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}\n")
            f.write(f"Total Volume: {data['Volume'].sum():,.0f}\n\n")
            
            # Write trend details
            f.write("Detailed Trend Analysis:\n")
            f.write("-" * 80 + "\n")
            f.write("Time          | Direction | Duration | Price Range      | Change    | Avg Volume\n")
            f.write("-" * 80 + "\n")
            
            for trend in trends:
                f.write(
                    f"{trend['start_time'].strftime('%H:%M')}-{trend['end_time'].strftime('%H:%M')} | "
                    f"{trend['direction']:<9} | {trend['duration']:>3}m     | "
                    f"${trend['start_price']:>7.2f}-${trend['end_price']:<7.2f} | "
                    f"${trend['price_change']:>7.2f} | {trend['avg_volume']:>9.0f}\n"
                )
            
            f.write("\nTrend Statistics:\n")
            f.write("-" * 40 + "\n")
            up_trends = [t for t in trends if t['direction'] == 'Up']
            down_trends = [t for t in trends if t['direction'] == 'Down']
            
            f.write(f"Uptrends: {len(up_trends)}\n")
            f.write(f"Downtrends: {len(down_trends)}\n")
            f.write(f"Average Trend Duration: {sum(t['duration'] for t in trends)/len(trends):.1f} minutes\n")
            f.write(f"Average Price Move: ${sum(abs(t['price_change']) for t in trends)/len(trends):.2f}\n")
            
            print(f"\nReport generated: {report_path}")

    def _generate_plots(self, data: pd.DataFrame, trends: list, symbol: str):
        """Generate analysis plots"""
        # Create plots directory if it doesn't exist
        plots_dir = Path('data/trend_analysis/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plot filename with date
        date_str = data.index[0].strftime('%Y%m%d')
        plot_path = plots_dir / f"trend_analysis_{symbol}_{date_str}.png"
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot price
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Price', color='black', linewidth=1)
        
        # Plot trends
        for trend in trends:
            trend_data = data[trend['start_time']:trend['end_time']]
            color = 'green' if trend['direction'] == 'Up' else 'red'
            plt.plot(trend_data.index, trend_data['Close'], color=color, linewidth=2)
        
        plt.title(f'Price Trends - {symbol} {date_str}')
        plt.ylabel('Price ($)')
        plt.grid(True)
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.bar(data.index, data['Volume'], color='blue', alpha=0.5)
        plt.title('Volume')
        plt.ylabel('Volume')
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot generated: {plot_path}")

    def _calculate_statistics(self, data: pd.DataFrame, trends: list) -> dict:
        """Calculate trend statistics"""
        return {
            'total_trends': len(trends),
            'up_trends': len([t for t in trends if t['direction'] == 'Up']),
            'down_trends': len([t for t in trends if t['direction'] == 'Down']),
            'avg_duration': sum(t['duration'] for t in trends) / len(trends),
            'avg_move': sum(abs(t['price_change']) for t in trends) / len(trends),
            'total_volume': data['Volume'].sum(),
            'price_range': data['High'].max() - data['Low'].min(),
            'day_change': data['Close'].iloc[-1] - data['Open'].iloc[0]
        }

    def _save_trend_data(self, data: pd.DataFrame, trends: list, symbol: str):
        """Save data with trend annotations"""
        # Create output directory if it doesn't exist
        output_dir = Path('data/stock_trend_complete')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create copy of data for annotation
        trend_data = data.copy()
        
        # Add trend column
        trend_data['Trend'] = 'None'
        
        # Annotate each trend
        for trend in trends:
            mask = (trend_data.index >= trend['start_time']) & (trend_data.index <= trend['end_time'])
            trend_data.loc[mask, 'Trend'] = trend['direction']
        
        # Generate output filename with date
        date_str = data.index[0].strftime('%Y%m%d')
        output_path = output_dir / f"{symbol}_trend_data_{date_str}.csv"
        
        # Save to CSV
        trend_data.to_csv(output_path)
        print(f"\nTrend data saved to: {output_path}")

# Usage example
if __name__ == "__main__":
    analyzer = TrendAnalysis()
    results = analyzer.analyze_data("data/stock_raw_data/NNE_data_20241205.csv")
    
    if results:
        print("\nAnalysis completed successfully")
        print(f"Trends identified: {len(results['trends'])}") 