import pandas as pd
import os
from datetime import datetime

def analyze_trends(input_file, output_file):
    """
    Analyzes trend data to identify Reversals and Countermoves
    
    Args:
        input_file (str): Path to preprocessed trend data CSV
        output_file (str): Path to save analysis results
        
    Raises:
        ValueError: If required columns are missing
        Exception: For other processing errors
    """
    try:
        # Load and validate data
        df = pd.read_csv(input_file, parse_dates=['Datetime'])
        required_columns = ['Datetime', 'Close', 'Trend']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Required: {required_columns}")

        # Initialize a new column for Reversal or Countermove
        df['Action'] = None

        # Iterate over the DataFrame to identify Reversals and Countermoves
        for i in range(1, len(df)):
            current_trend = df.loc[i, 'Trend']
            previous_trend = df.loc[i - 1, 'Trend']
            current_close = df.loc[i, 'Close']
            previous_close = df.loc[i - 1, 'Close']

            # Identify Reversal Points
            if (current_trend == 'UpTrend' and previous_trend == 'DownTrend') or \
               (current_trend == 'DownTrend' and previous_trend == 'UpTrend'):
                df.loc[i, 'Action'] = 'Reversal'

            # Identify Countermoves
            elif current_trend == previous_trend:
                if (current_trend == 'UpTrend' and current_close < previous_close) or \
                   (current_trend == 'DownTrend' and current_close > previous_close):
                    df.loc[i, 'Action'] = 'Countermove'

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)

        print(f"Analysis complete. Results saved to {output_file}")
        return True

    except Exception as e:
        print(f"Error analyzing {input_file}: {str(e)}")
        raise

def process_all_files(base_path, output_base_path):
    """
    Process all preprocessed trend files in the directory
    
    Args:
        base_path (str): Directory containing input files
        output_base_path (str): Directory for processed files
        
    Returns:
        int: Number of files successfully processed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    files_processed = 0
    errors = []
    
    for file in os.listdir(base_path):
        if file.startswith('trend_analysis_pp_NNE_') and file.endswith('.csv'):
            try:
                # Extract date from filename
                date = file.split('_')[4].split('.')[0]
                
                # Define input and output paths
                input_file = os.path.join(base_path, file)
                output_file = os.path.join(output_base_path, f'counter_riversal_NNE_{date}.csv')
                
                print(f"\nProcessing file: {file}")
                analyze_trends(input_file, output_file)
                files_processed += 1
                
            except Exception as e:
                error_msg = f"Error processing {file}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total files processed: {files_processed}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        print("Error details:")
        for error in errors:
            print(f"- {error}")
    
    return files_processed

if __name__ == "__main__":
    # Define base paths
    base_path = r'D:\NNE_strategy\nne_strategy\data\preprocess_trend_data'
    output_base_path = r'D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis'
    
    print("\nSelect processing mode:")
    print("1. Process single file")
    print("2. Process all files")
    mode = input("Enter your choice (1 or 2): ")
    
    if mode == "1":
        # Get the date from user input
        date_input = input("Enter the date for analysis (YYYYMMDD): ")
        
        try:
            # Validate date format
            datetime.strptime(date_input, '%Y%m%d')
            
            # Define file paths
            input_file = os.path.join(base_path, f'trend_analysis_pp_NNE_{date_input}.csv')
            output_file = os.path.join(output_base_path, f'counter_riversal_NNE_{date_input}.csv')
            
            # Check if input file exists
            if os.path.exists(input_file):
                analyze_trends(input_file, output_file)
            else:
                print(f"Error: File not found for date {date_input}")
                
        except ValueError:
            print("Error: Invalid date format. Please use YYYYMMDD format.")
    
    elif mode == "2":
        # Process all files
        files_processed = process_all_files(base_path, output_base_path)
        print(f"\nBatch processing completed. Total files processed: {files_processed}")
    
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.") 