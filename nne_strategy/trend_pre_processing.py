import pandas as pd
import os
from datetime import datetime

def preprocess_trend_data(input_file, output_file, report_file):
    """Process a single file with trend data"""
    # Load the data
    df = pd.read_csv(input_file, parse_dates=['Datetime'])

    # Initialize a report
    report = []

    # Check for missing values
    missing_before = df.isnull().sum()
    report.append("Missing values before interpolation:\n" + str(missing_before) + "\n")

    # Interpolate missing values for numerical columns
    df.interpolate(method='linear', inplace=True)

    # Fill missing Trend values using forward fill
    df['Trend'].fillna(method='ffill', inplace=True)

    # Check for missing values after interpolation and filling
    missing_after = df.isnull().sum()
    report.append("Missing values after interpolation and filling:\n" + str(missing_after) + "\n")

    # Save the processed data to a new CSV file
    df.to_csv(output_file, index=False)

    # Write the report to a text file
    with open(report_file, 'w') as f:
        f.write("\n".join(report))

    print(f"Pre-processing complete. Processed data saved to {output_file} and report saved to {report_file}.")

def process_all_files(base_path, output_base_path):
    """Process all trend analysis files in the directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    # Get all CSV files in the base path
    files_processed = 0
    for file in os.listdir(base_path):
        if file.startswith('trend_analysis_NNE_') and file.endswith('.csv'):
            try:
                # Extract date from filename
                date = file.split('_')[3].split('.')[0]  # Gets YYYYMMDD from filename
                
                # Define input and output paths
                input_file = os.path.join(base_path, file)
                output_file = os.path.join(output_base_path, f'trend_analysis_pp_NNE_{date}.csv')
                report_file = os.path.join(output_base_path, f'trend_analysis_pp_NNE_{date}_report.txt')
                
                # Process the file
                print(f"\nProcessing file: {file}")
                preprocess_trend_data(input_file, output_file, report_file)
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    print(f"\nCompleted processing {files_processed} files.")
    return files_processed

if __name__ == "__main__":
    # Define base paths
    base_path = r'D:\NNE_strategy\nne_strategy\data\stock_trend_complete'
    output_base_path = r'D:\NNE_strategy\nne_strategy\data\preprocess_trend_data'
    
    # Ask user for processing mode
    print("\nSelect processing mode:")
    print("1. Process single file")
    print("2. Process all files")
    mode = input("Enter your choice (1 or 2): ")
    
    if mode == "1":
        # Get the date from user input
        date_input = input("Enter the date for the preprocessed file (YYYYMMDD): ")
        
        # Define file paths
        input_file = os.path.join(base_path, f'trend_analysis_NNE_{date_input}.csv')
        output_file = os.path.join(output_base_path, f'trend_analysis_pp_NNE_{date_input}.csv')
        report_file = os.path.join(output_base_path, f'trend_analysis_pp_NNE_{date_input}_report.txt')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_base_path, exist_ok=True)
        
        # Run the pre-processing
        if os.path.exists(input_file):
            preprocess_trend_data(input_file, output_file, report_file)
        else:
            print(f"Error: File not found for date {date_input}")
    
    elif mode == "2":
        # Process all files
        files_processed = process_all_files(base_path, output_base_path)
        print(f"Total files processed: {files_processed}")
    
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")
