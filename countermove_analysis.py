import pandas as pd
from datetime import datetime
import os
from pathlib import Path

def analyze_all_countermoves(file_path):
    """
    Analyzes all counter-moves within each trend, identifying their characteristics and patterns.
    
    Args:
        file_path (str): Path to the CSV file containing trend analysis data
    """
    try:
        # Read and prepare data
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        analysis_date = df['Datetime'].iloc[0].strftime('%Y%m%d')
        
        # Initialize lists to store trends and counter-moves
        trends = []
        current_trend = None
        
        # Process each row to identify trends and counter-moves
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check for trend start
            if current_row['Trend_Start']:
                if current_trend:
                    # Complete previous trend
                    current_trend['end_time'] = prev_row['Datetime']
                    current_trend['end_price'] = prev_row['Close']
                    current_trend['duration'] = current_trend['end_time'] - current_trend['start_time']
                    current_trend['total_move'] = current_trend['end_price'] - current_trend['start_price']
                    trends.append(current_trend)
                
                # Start new trend
                current_trend = {
                    'direction': current_row['Trend'],
                    'start_time': current_row['Datetime'],
                    'start_price': current_row['Open'],
                    'counter_moves': []
                }
            
            # Detect counter-moves within current trend
            elif current_trend:
                price_change = current_row['Close'] - prev_row['Close']
                
                # Counter-move detection logic
                if ((current_trend['direction'] == 'UpTrend' and price_change < 0) or 
                    (current_trend['direction'] == 'DownTrend' and price_change > 0)):
                    
                    counter_move = {
                        'time': current_row['Datetime'],
                        'size': abs(price_change),
                        'price_range': current_row['High'] - current_row['Low'],
                        'open': current_row['Open'],
                        'high': current_row['High'],
                        'low': current_row['Low'],
                        'close': current_row['Close'],
                        'prev_close': prev_row['Close'],
                        'prev_range': prev_row['High'] - prev_row['Low']
                    }
                    current_trend['counter_moves'].append(counter_move)
        
        # Complete final trend
        if current_trend:
            current_trend['end_time'] = df.iloc[-1]['Datetime']
            current_trend['end_price'] = df.iloc[-1]['Close']
            current_trend['duration'] = current_trend['end_time'] - current_trend['start_time']
            current_trend['total_move'] = current_trend['end_price'] - current_trend['start_price']
            trends.append(current_trend)
        
        # Update output directory path
        output_dir = Path(__file__).parent / "data" / "countermove_dataset"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            
        output_file = output_dir / f"detailed_countermove_analysis_{analysis_date}.txt"
        
        with open(output_file, 'w') as f:
            write_analysis_to_file(f, trends, analysis_date)
            
        print(f"\nDetailed analysis has been saved to: {output_file}")
        return trends
        
    except Exception as e:
        print(f"Error analyzing counter-moves: {e}")
        return None

def write_analysis_to_file(f, trends, analysis_date):
    """Write the analysis results to the output file"""
    f.write("DETAILED COUNTER-MOVE ANALYSIS\n")
    f.write(f"Date: {analysis_date}\n")
    f.write("============================\n\n")
    
    for i, trend in enumerate(trends, 1):
        # Write trend information with clear header
        f.write(f"MAJOR TREND #{i}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Direction: {trend['direction']}\n")
        f.write(f"Start: {trend['start_time'].strftime('%H:%M')} @ ${trend['start_price']:.2f}\n")
        f.write(f"End: {trend['end_time'].strftime('%H:%M')} @ ${trend['end_price']:.2f}\n")
        f.write(f"Duration: {trend['duration']}\n")
        f.write(f"Total Move: ${trend['total_move']:.2f}\n\n")
        
        # Write counter-moves details with clear separation
        if trend['counter_moves']:
            f.write("COUNTER-MOVES WITHIN THIS TREND:\n")
            f.write("-" * 50 + "\n")
            
            for j, cm in enumerate(trend['counter_moves'], 1):
                percentage = (cm['size'] / abs(trend['total_move'])) * 100
                f.write(f"Counter-move #{j}\n")
                f.write(f"Time: {cm['time'].strftime('%H:%M')}\n")
                f.write(f"Counter-move size: ${cm['size']:.2f}\n")
                f.write(f"Percentage of trend: {percentage:.1f}%\n")
                f.write(f"Price range: ${cm['price_range']:.2f}\n")
                f.write(f"Open: ${cm['open']:.2f}\n")
                f.write(f"High: ${cm['high']:.2f}\n")
                f.write(f"Low: ${cm['low']:.2f}\n")
                f.write(f"Close: ${cm['close']:.2f}\n")
                f.write(f"Previous candle close: ${cm['prev_close']:.2f}\n")
                f.write(f"Previous candle range: ${cm['prev_range']:.2f}\n")
                f.write("-" * 30 + "\n\n")
        else:
            f.write("NO COUNTER-MOVES DETECTED IN THIS TREND\n\n")
        
        f.write("=" * 80 + "\n\n")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime

    # Get date from command line or use default
    if len(sys.argv) > 1:
        try:
            # Convert YYYYMMDD to YYYY-MM-DD
            input_date = sys.argv[1]
            analysis_date = input_date  # Keep YYYYMMDD format for filenames
            DATE = f"{input_date[:4]}-{input_date[4:6]}-{input_date[6:]}"  # Convert to YYYY-MM-DD for display
        except Exception as e:
            print(f"Error parsing date argument: {e}")
            print("Please use format YYYYMMDD (e.g., 20241209)")
            sys.exit(1)
    else:
        # Use current date if no arguments provided
        DATE = datetime.now().strftime('%Y-%m-%d')
        analysis_date = datetime.now().strftime('%Y%m%d')

    # Set up the file path
    script_dir = Path(__file__).parent
    csv_file = script_dir / "data" / "stock_trend_complete" / f"trend_analysis_NNE_{analysis_date}.csv"
    
    if not csv_file.exists():
        print(f"Error: Could not find CSV file at: {csv_file}")
        print("Please ensure the trend analysis file exists and the path is correct.")
        sys.exit(1)
        
    print(f"\nAnalyzing countermoves for NNE on {DATE}")
    print("=" * 50)
    
    analyze_all_countermoves(str(csv_file))