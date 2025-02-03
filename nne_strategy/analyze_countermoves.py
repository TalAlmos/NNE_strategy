import pandas as pd
import numpy as np
import json
import glob
import os

def read_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV into a pandas DataFrame and ensure Datetime is parsed.
    """
    df = pd.read_csv(csv_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def find_countermoves(df: pd.DataFrame):
    """
    Identify trend changes and classify them as Countermoves or Reversals.
    A trend change occurs when the 'Trend' value changes from the previous row.
    
    Returns a DataFrame with a new 'Action' column indicating:
    - 'Countermove': Short-term trend changes
    - 'Reversal': Significant trend changes that persist
    """
    # Initialize the Action column
    df['Action'] = None
    
    # Detect trend changes
    trend_changes = df['Trend'] != df['Trend'].shift()
    
    # Initial classification of all trend changes as Countermoves
    df.loc[trend_changes, 'Action'] = 'Countermove'
    
    # Look for sustained trend changes (Reversals)
    # For example, if a trend change persists for more than N periods
    REVERSAL_THRESHOLD = 5  # Adjust this value based on your requirements
    
    for i in df[trend_changes].index:
        if i + REVERSAL_THRESHOLD >= len(df):
            continue
            
        current_trend = df.loc[i, 'Trend']
        future_trends = df.loc[i:i+REVERSAL_THRESHOLD, 'Trend']
        
        # If the new trend persists, classify as Reversal
        if all(future_trends == current_trend):
            df.loc[i, 'Action'] = 'Reversal'
    
    return df

def analyze_countermove_segments(df: pd.DataFrame, segments: list) -> pd.DataFrame:
    """
    For each tuple (start_idx, end_idx), compute:
      - StartTime, EndTime
      - Duration (in minutes) as simply (end_idx - start_idx + 1) 
        when each row is a 1-minute candle
      - PriceAction (end_close - start_open)
      - PricePct (% change from the segment's starting Open)
      - Volume (sum of volumes)
    """
    result = []

    for (start, end) in segments:
        segment_df = df.loc[start:end].copy()
        start_time = segment_df.iloc[0]['Datetime']
        end_time = segment_df.iloc[-1]['Datetime']

        # Row-based duration for 1-minute bars
        duration_minutes = (end - start + 1)

        start_open = segment_df.iloc[0]['Open']
        end_close = segment_df.iloc[-1]['Close']
        price_action = end_close - start_open
        price_pct = np.nan
        if start_open and (start_open != 0):
            price_pct = (price_action / start_open) * 100.0

        volume_sum = segment_df['Volume'].sum()

        result.append({
            'StartTime': start_time.isoformat(),
            'EndTime': end_time.isoformat(),
            'Duration': duration_minutes,
            'PriceAction': price_action,
            'PricePct': price_pct,
            'Volume': volume_sum
        })

    return pd.DataFrame(result)

def categorize_countermoves(countermove_df: pd.DataFrame, group_by='PricePct'):
    """
    Assign each row to a 'Small', 'Medium', or 'Large' group based on quantiles
    of the chosen metric. Then compute group-level average stats.

    group_by can be 'PricePct', 'Duration', or any numeric column.
    By default, we use absolute values for the grouping metric so that 
    negative vs positive moves are grouped by magnitude.
    """
    metric = countermove_df[group_by].abs()
    q33 = metric.quantile(0.33)
    q66 = metric.quantile(0.66)

    countermove_df['SizeGroup'] = countermove_df[group_by].apply(
        lambda val: 'Small' if abs(val) <= q33 else ('Medium' if abs(val) <= q66 else 'Large')
    )

    grouped = countermove_df.groupby('SizeGroup').agg({
        'Duration': 'mean',
        'PriceAction': 'mean',
        'PricePct': 'mean',
        'Volume': 'mean'
    }).rename(columns={
        'Duration': 'AvgDuration',
        'PriceAction': 'AvgPriceAction',
        'PricePct': 'AvgPricePct',
        'Volume': 'AvgVolume'
    }).reset_index()

    return countermove_df, grouped

def save_analysis_results(pos_grouped_stats, neg_grouped_stats, output_path):
    # Format the output to include both positive and negative group labels in the JSON
    results = {
        'positive': pos_grouped_stats.to_dict(orient='records'),
        'negative': neg_grouped_stats.to_dict(orient='records')
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def process_file(input_file: str, output_file: str):
    """
    Process a single trend analysis file to identify countermoves and reversals.
    
    Args:
        input_file: Path to the input CSV file (from preprocess_trend_data)
        output_file: Path where to save the processed file
    """
    try:
        # Read the input file
        df = pd.read_csv(input_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Add countermove/reversal classifications
        df = find_countermoves(df)
        
        # Save the enhanced dataset
        df.to_csv(output_file, index=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return False

def main():
    """
    Process all files in the preprocess_trend_data directory
    """
    input_dir = r"D:\NNE_strategy\nne_strategy\data\preprocess_trend_data"
    output_dir = r"D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "trend_analysis_pp_*.csv"))
    
    for input_file in input_files:
        # Generate output filename
        base_name = os.path.basename(input_file)
        output_name = base_name.replace("trend_analysis_pp_", "counter_riversal_")
        output_file = os.path.join(output_dir, output_name)
        
        # Process the file
        success = process_file(input_file, output_file)
        if success:
            print(f"Successfully processed {base_name}")
        else:
            print(f"Failed to process {base_name}")

if __name__ == '__main__':
    main()
