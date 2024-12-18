from pathlib import Path
from trend_finder_v3 import TrendFinder
import pandas as pd
import sys

def analyze_all_data():
    # Get raw data directory
    raw_data_dir = Path(__file__).parent / "data" / "stock_raw_data"
    output_dir = Path(__file__).parent / "data" / "stock_trend_complete"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all NNE data files
    data_files = list(raw_data_dir.glob("NNE_data_*.csv"))
    
    print(f"\nFound {len(data_files)} data files to analyze")
    print("=" * 50)
    
    # Process each file
    for data_file in sorted(data_files):
        date = data_file.stem.split('_')[-1]  # Extract date from filename
        print(f"\nProcessing {date}...")
        
        try:
            # Load data
            data = pd.read_csv(data_file)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Initialize and run trend finder
            finder = TrendFinder()
            summary = finder.detect_trend(data)
            
            if summary:
                # Save results to CSV
                trends_df = pd.DataFrame(finder.trends)
                trends_df.to_csv(output_dir / f"trend_analysis_v3_NNE_{date}.csv", index=False)
                
                # Save summary to TXT
                with open(output_dir / f"trend_analysis_v3_NNE_{date}_summary.txt", 'w') as f:
                    f.write(f"NNE Trend Analysis Summary (v3) for {date}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Total Trends: {summary['total_trends']}\n")
                    f.write(f"First Point: {summary['first_point']['time'].strftime('%H:%M')} @ ${summary['first_point']['price']:.2f}\n")
                    f.write(f"Last Point: {summary['last_point']['time'].strftime('%H:%M')} @ ${summary['last_point']['price']:.2f}\n\n")
                    
                    f.write("Significant Trends:\n")
                    f.write("-" * 30 + "\n")
                    for i, trend in enumerate(finder.trends, 1):
                        f.write(f"\nTrend #{i}:\n")
                        f.write(f"Time: {trend['start_time'].strftime('%H:%M')} -> {trend['end_time'].strftime('%H:%M')}\n")
                        f.write(f"Price: ${trend['start_price']:.2f} -> ${trend['end_price']:.2f}\n")
                        f.write(f"Direction: {trend['direction']}\n")
                        f.write(f"Change: ${trend['price_change']:.2f}\n")
                        f.write(f"Duration: {trend['duration']:.0f} minutes\n")
                        f.write(f"Significance: {trend['significance']:.2f}\n")
                
                print(f"Analysis complete for {date}")
            else:
                print(f"Failed to analyze {date}")
                
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue
    
    print("\nAnalysis complete for all files")

if __name__ == "__main__":
    analyze_all_data() 