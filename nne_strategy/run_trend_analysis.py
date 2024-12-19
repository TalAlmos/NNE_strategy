from trend_analysis import TrendAnalysis
import pandas as pd
from pathlib import Path
import glob
import os

def get_latest_data_file(data_dir):
    # Get list of all data files
    files = glob.glob(str(data_dir / "NNE_data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    # Return the most recent file
    return max(files, key=os.path.getctime)

def main():
    try:
        # Initialize analyzer
        analyzer = TrendAnalysis()
        
        # Get absolute paths
        workspace_root = Path("D:/NNE_strategy")
        raw_data_dir = workspace_root / "nne_strategy/data/raw"
        output_dir = workspace_root / "nne_strategy/data/analysis/trends"
        
        print(f"Using workspace root: {workspace_root}")
        print(f"Raw data directory: {raw_data_dir}")
        print(f"Output directory: {output_dir}")
        
        # Get the latest data file
        input_file = get_latest_data_file(raw_data_dir)
        print(f"Loading data from: {input_file}")
        
        # Load and analyze data
        data = pd.read_csv(input_file)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        
        # Run trend analysis
        print("Running trend analysis...")
        results = analyzer.identify_trends(data)
        
        # Create output directory
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename based on input filename
        input_date = Path(input_file).stem.split('_')[-1]
        output_file = output_dir / f"trend_analysis_detailed_{input_date}.csv"
        print(f"Saving results to: {output_file}")
        
        # Save results
        results.to_csv(output_file, index=False)
        
        print(f"\nTrend Analysis Results:")
        print("=" * 50)
        print(f"Analyzed file: {Path(input_file).name}")
        print(f"Total Bars: {len(results)}")
        print(f"Up Trends: {len(results[results['Trend'] == 'Up'])}")
        print(f"Down Trends: {len(results[results['Trend'] == 'Down'])}")
        print(f"Reversals: {len(results[results['IsReversal']])}")
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 