import pandas as pd
import os

def preprocess_trend_data(input_file, output_file, report_file):
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

if __name__ == "__main__":
    # Define file paths
    base_path = r'D:\NNE_strategy\nne_strategy\data\stock_trend_complete'
    input_file = os.path.join(base_path, 'trend_analysis_NNE_20241205.csv')
    output_file = os.path.join(base_path, 'trend_analysis_pp_NNE_20241205.csv')
    report_file = os.path.join(base_path, 'trend_analysis_pp_NNE_20241205_report.txt')

    # Run the pre-processing
    preprocess_trend_data(input_file, output_file, report_file)
