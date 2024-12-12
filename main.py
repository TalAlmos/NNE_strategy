import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def analyze_specific_countermove(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Look at a window around 10:43
        start_time = pd.to_datetime('2023-12-06 10:40:00')  # Changed year to 2023
        end_time = pd.to_datetime('2023-12-06 10:45:00')
        
        time_window = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
        
        print("\nPrice movement around 10:43:")
        for idx, row in time_window.iterrows():
            print(f"Time: {row['Datetime'].strftime('%H:%M')} | "
                  f"Open: ${row['Open']:.2f} | "
                  f"High: ${row['High']:.2f} | "
                  f"Low: ${row['Low']:.2f} | "
                  f"Close: ${row['Close']:.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing counter-move: {e}")
        return None

# Main execution
csv_file_path = r"D:\NNE_strategy\NNE_data_20241206.csv"
analysis = analyze_specific_countermove(csv_file_path)