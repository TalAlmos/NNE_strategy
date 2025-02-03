import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import os

# Define paths globally
input_folder = r'D:\NNE_strategy\nne_strategy\data\raw'
output_folder = r'D:\NNE_strategy\nne_strategy\data\stock_trend_complete'

# Create directories
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

def ensure_min_max_alternating(selected_points):
    # Make sure min and max are alternating, discarding consecutive min or max
    selected_points['type'] = selected_points.apply(
        lambda row: 'min' if not pd.isna(row['min']) else 'max' if not pd.isna(row['max']) else 'none',
        axis=1
    )

    # Filter to remove consecutive 'min' or 'max' rows
    filtered_rows = []
    last_type = None
    for _, row in selected_points.iterrows():
        if row['type'] != last_type and row['type'] != 'none':
            filtered_rows.append(row)
            last_type = row['type']

    # Create a new DataFrame with the filtered rows
    return pd.DataFrame(filtered_rows).drop(columns=['type'])

def process_all_files(input_folder, output_folder):
    """Process all CSV files in the input folder"""
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            input_file = os.path.join(input_folder, file)
            print(f"Processing file: {input_file}")
            
            # Load and process the data
            df = pd.read_csv(input_file)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Extract the date from the first entry in the 'Datetime' column
            data_date = df['Datetime'].dt.date.iloc[0].strftime('%Y%m%d')
            
            # Find local minima and maxima
            n = 2  # Number of points to consider for local extrema
            df['min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']
            df['max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']
            
            # Add first point (09:30)
            first_point = df.iloc[0]
            # Find next extremum point to determine type
            next_min_idx = df[pd.notna(df['min'])].index[0] if len(df[pd.notna(df['min'])]) > 0 else None
            next_max_idx = df[pd.notna(df['max'])].index[0] if len(df[pd.notna(df['max'])]) > 0 else None
            
            if next_min_idx is not None and next_max_idx is not None:
                if next_min_idx < next_max_idx:  # Next point is a minimum
                    df.loc[0, 'max'] = first_point['Close']  # Set first point as maximum
                else:  # Next point is a maximum
                    df.loc[0, 'min'] = first_point['Close']  # Set first point as minimum
            
            # Add last point (15:59)
            last_point = df.iloc[-1]
            # Find previous extremum point to determine type
            prev_min_idx = df[pd.notna(df['min'])].index[-1] if len(df[pd.notna(df['min'])]) > 0 else None
            prev_max_idx = df[pd.notna(df['max'])].index[-1] if len(df[pd.notna(df['max'])]) > 0 else None
            
            if prev_min_idx is not None and prev_max_idx is not None:
                if prev_min_idx > prev_max_idx:  # Previous point is a minimum
                    df.loc[len(df)-1, 'max'] = last_point['Close']  # Set last point as maximum
                else:  # Previous point is a maximum
                    df.loc[len(df)-1, 'min'] = last_point['Close']  # Set last point as minimum
            
            # Combine minima and maxima into a single DataFrame
            extrema = pd.concat([df[['Datetime', 'min']].dropna(), df[['Datetime', 'max']].dropna()])
            extrema.sort_index(inplace=True)
            
            # Group by hour and select the highest maxima and lowest minima
            extrema['Hour'] = extrema['Datetime'].dt.floor('h')
            hourly_maxima = extrema.groupby('Hour')['max'].idxmax().dropna()
            hourly_minima = extrema.groupby('Hour')['min'].idxmin().dropna()
            
            # Combine selected points
            selected_points = extrema.loc[pd.concat([hourly_maxima, hourly_minima])].sort_index()
            
            # Add additional points
            additional_points = extrema.iloc[[len(extrema)//3, 2*len(extrema)//3]]
            selected_points = pd.concat([selected_points, additional_points]).sort_index()
            
            # Make sure min and max are alternating
            selected_points = ensure_min_max_alternating(selected_points)
            
            # Ensure distance between points is greater than 5 minutes
            selected_points = selected_points.loc[selected_points['Datetime'].diff().fillna(pd.Timedelta(minutes=6)) > pd.Timedelta(minutes=5)]
            
            # Prepare trend summary
            trends = []
            for i in range(len(selected_points) - 1):
                start = selected_points.iloc[i]
                end = selected_points.iloc[i + 1]
                if pd.notna(start['min']) and pd.notna(end['max']):
                    trend_type = 'UpTrend'
                    start_price = start['min']
                    end_price = end['max']
                elif pd.notna(start['max']) and pd.notna(end['min']):
                    trend_type = 'DownTrend'
                    start_price = start['max']
                    end_price = end['min']
                else:
                    continue
                
                trends.append({
                    'Trend Type': trend_type,
                    'Start Price': start_price,
                    'Start Time': start['Datetime'],
                    'End Price': end_price,
                    'End Time': end['Datetime']
                })
            
            # Add trend indications to the DataFrame
            df['Trend'] = None
            for trend in trends:
                mask = (df['Datetime'] >= trend['Start Time']) & (df['Datetime'] <= trend['End Time'])
                df.loc[mask, 'Trend'] = trend['Trend Type']
            
            # Save the processed data
            output_file = os.path.join(output_folder, f'trend_analysis_NNE_{data_date}.csv')
            df.to_csv(output_file, index=False)
            
            # Generate plot
            plt.figure(figsize=(14, 7))
            plt.plot(df['Datetime'], df['Close'], label='Close Price', color='blue')
            plt.scatter(selected_points['Datetime'], selected_points['min'], label='Local Minima', color='red')
            plt.scatter(selected_points['Datetime'], selected_points['max'], label='Local Maxima', color='green')
            
            # Draw trend lines
            for i in range(len(selected_points) - 1):
                if (not np.isnan(selected_points['min'].iloc[i]) and not np.isnan(selected_points['max'].iloc[i+1])) or \
                   (not np.isnan(selected_points['max'].iloc[i]) and not np.isnan(selected_points['min'].iloc[i+1])):
                    plt.plot(selected_points['Datetime'].iloc[i:i+2], 
                            selected_points[['min', 'max']].iloc[i:i+2].mean(axis=1), 
                            color='orange', linewidth=2)
            
            plt.title(f'Stock Price with Trend Lines - {data_date}')
            plt.xlabel('Datetime')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(output_folder, f'trend_analysis_plot_NNE_{data_date}.png'))
            plt.close()
            
            print(f"Completed processing file: {file}")

if __name__ == "__main__":
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    process_all_files(input_folder, output_folder)