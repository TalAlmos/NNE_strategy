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
    Identify trend changes (Reversals) and countermoves using Close prices.
    
    Rules:
    1. Skip rows with missing Trend values for both Reversals and Countermoves
    2. Skip first and last rows for Reversals
    3. Only compare prices with previous rows that have valid Trend values
    """
    # Initialize the Action column if not present
    if 'Action' not in df.columns:
        df['Action'] = None

    # First mark Reversals at trend changes
    trend_changes = df['Trend'] != df['Trend'].shift()
    
    # Apply rules for Reversals
    for i in range(1, len(df) - 1):  # Skip first and last rows
        # Skip if current or previous Trend is missing
        if pd.isna(df.loc[i, 'Trend']) or pd.isna(df.loc[i-1, 'Trend']):
            continue
            
        # Mark Reversal only if there's a valid trend change
        if trend_changes.iloc[i]:
            df.loc[i, 'Action'] = 'Reversal'

    # Now identify Countermoves based on Close prices
    for i in range(1, len(df)):
        if df.loc[i, 'Action'] == 'Reversal':  # Skip reversal points
            continue
            
        current_trend = df.loc[i, 'Trend']
        # Skip if current trend is missing
        if pd.isna(current_trend):
            continue
            
        # Find the previous valid row (with Trend indication)
        prev_idx = i - 1
        while prev_idx >= 0 and pd.isna(df.loc[prev_idx, 'Trend']):
            prev_idx -= 1
            
        # Skip if no valid previous row found
        if prev_idx < 0:
            continue
            
        prev_close = df.loc[prev_idx, 'Close']
        current_close = df.loc[i, 'Close']
        
        if current_trend == 'UpTrend':
            # In uptrend, mark lower closes as countermoves
            if current_close < prev_close:
                df.loc[i, 'Action'] = 'Countermove'
                
        elif current_trend == 'DownTrend':
            # In downtrend, mark higher closes as countermoves
            if current_close > prev_close:
                df.loc[i, 'Action'] = 'Countermove'

    return df

def group_and_analyze_countermoves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group consecutive 'Countermove' rows and calculate statistics.
    
    Measurements for each countermove:
    1. Duration: Number of bars in the countermove
    2. Price Change: Total price movement from start to end
    3. Percentage Change: Price change as a percentage
    4. Volume: Total volume during the countermove
    5. Start/End Times: When the countermove began and ended
    
    Args:
        df: DataFrame with 'Action' and 'Close' columns
    
    Returns:
        DataFrame with countermove statistics
    """
    if 'Action' not in df.columns:
        return pd.DataFrame()

    # Initialize lists to store countermove data
    countermoves = []
    
    # Find rows marked as Countermove
    countermove_mask = df['Action'] == 'Countermove'
    
    if not countermove_mask.any():
        return pd.DataFrame()

    # Group consecutive Countermove rows
    for _, group in df[countermove_mask].groupby((countermove_mask != countermove_mask.shift()).cumsum()):
        if len(group) > 0:
            # Find the row before the countermove started (for reference price)
            start_idx = group.index[0]
            if start_idx > 0:
                ref_idx = start_idx - 1
                ref_price = df.loc[ref_idx, 'Close']
                
                # Calculate statistics for this countermove
                start_time = group.iloc[0]['Datetime']
                end_time = group.iloc[-1]['Datetime']
                duration = len(group)  # Number of bars
                
                # Price changes
                start_price = ref_price  # Reference price before countermove
                end_price = group.iloc[-1]['Close']
                price_change = end_price - start_price
                price_pct = (price_change / start_price) * 100
                
                # Volume
                volume = group['Volume'].sum()
                
                # Determine direction based on trend
                trend = group.iloc[0]['Trend']
                
                countermoves.append({
                    'StartTime': start_time,
                    'EndTime': end_time,
                    'Duration': duration,
                    'StartPrice': start_price,
                    'EndPrice': end_price,
                    'PriceChange': price_change,
                    'PriceChangePct': price_pct,
                    'Volume': volume,
                    'Trend': trend
                })

    # Convert to DataFrame
    countermoves_df = pd.DataFrame(countermoves)
    
    return countermoves_df

def categorize_countermoves(df: pd.DataFrame, group_by: str = 'PricePct'):
    """
    Categorize countermoves into size groups based on their price action or percentage.
    Returns both the categorized DataFrame and summary statistics in JSON-serializable format.
    """
    if df.empty:
        return df, []
        
    # Calculate quartiles for grouping
    q1 = df[group_by].quantile(0.25)
    q3 = df[group_by].quantile(0.75)
    
    # Create size categories
    conditions = [
        (df[group_by] >= q3),
        (df[group_by] < q1),
    ]
    choices = ['Large', 'Small']
    df['SizeGroup'] = np.select(conditions, choices, default='Medium')
    
    # Calculate statistics for each size group
    grouped_stats = []
    for group_name, group_data in df.groupby('SizeGroup'):
        stats = {
            'SizeGroup': group_name,
            'AvgDuration': float(group_data['Duration'].mean()),
            'AvgPriceAction': float(group_data['PriceAction'].mean()),
            'AvgPricePct': float(group_data['PricePct'].mean()),
            'AvgVolume': float(group_data['Volume'].mean())
        }
        grouped_stats.append(stats)
    
    return df, grouped_stats

def save_analysis_results(pos_grouped_stats, neg_grouped_stats, output_path):
    """
    Save the positive and negative countermove stats as JSON.
    """
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

def process_and_analyze_files(input_dir: str, stats_output_dir: str):
    """
    Process files and save both processed CSVs and consolidated statistics.
    """
    # Create output directories if they don't exist
    os.makedirs(stats_output_dir, exist_ok=True)
    processed_dir = os.path.join(os.path.dirname(stats_output_dir), "processed_trend_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Initialize consolidated DataFrames
    all_countermoves_df = pd.DataFrame()
    
    # Get all CSV files in the input directory
    input_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_files.append(os.path.join(root, file))
    
    print(f"Found {len(input_files)} CSV files to process.")
    
    for input_file in input_files:
        try:
            # Read and process the file
            df = read_data(input_file)
            df = find_countermoves(df)
            
            # Save processed file with countermoves and reversals
            base_name = os.path.basename(input_file)
            processed_name = f"processed_{base_name}"
            processed_path = os.path.join(processed_dir, processed_name)
            df.to_csv(processed_path, index=False)
            print(f"Saved processed file: {processed_name}")
            
            # Analyze countermoves
            countermoves_df = group_and_analyze_countermoves(df)
            if not countermoves_df.empty:
                # Add source file information
                countermoves_df['Source'] = base_name
                all_countermoves_df = pd.concat([all_countermoves_df, countermoves_df])
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue
    
    # Generate consolidated statistics
    if not all_countermoves_df.empty:
        # Split by trend
        uptrend_moves = all_countermoves_df[all_countermoves_df['Trend'] == 'UpTrend']
        downtrend_moves = all_countermoves_df[all_countermoves_df['Trend'] == 'DownTrend']
        
        # Calculate statistics for each trend type
        consolidated_stats = {
            "summary": {
                "total_files_processed": len(input_files),
                "date_range": {
                    "start": all_countermoves_df['StartTime'].min(),
                    "end": all_countermoves_df['EndTime'].max()
                },
                "total_countermoves": {
                    "uptrend": len(uptrend_moves),
                    "downtrend": len(downtrend_moves)
                }
            },
            "uptrend_countermoves": uptrend_moves.describe().to_dict(),
            "downtrend_countermoves": downtrend_moves.describe().to_dict()
        }
        
        # Save consolidated statistics
        json_path = os.path.join(stats_output_dir, "consolidated_countermove_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(consolidated_stats, f, indent=4, default=str)
        
        print(f"\nConsolidated statistics saved to: {json_path}")
        return consolidated_stats
    else:
        print("No countermoves found in any files.")
        return None

def main():
    """
    Process all files and generate consolidated statistics
    """
    input_dir = r"D:\NNE_strategy\nne_strategy\data\trend"
    stats_dir = r"D:\NNE_strategy\nne_strategy\data\stats"
    
    print(f"Starting analysis of files in: {input_dir}")
    print(f"Results will be saved to: {stats_dir}")
    
    stats = process_and_analyze_files(input_dir, stats_dir)
    
    if stats:
        print("\nProcessing complete! Summary:")
        print(f"Total files processed: {stats['summary']['total_files_processed']}")
        print(f"Date range: {stats['summary']['date_range']['start']} to {stats['summary']['date_range']['end']}")
        print(f"Total countermoves analyzed: {stats['summary']['total_countermoves']['uptrend'] + stats['summary']['total_countermoves']['downtrend']}")
    
if __name__ == '__main__':
    main()
