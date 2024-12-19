"""
Process raw data files and perform trend analysis
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Optional, List
import sys
import matplotlib.pyplot as plt

from nne_strategy.trend_analysis import TrendAnalysis
from nne_strategy.config.config import config

logger = logging.getLogger(__name__)

def process_raw_file(file_path: Path, analyzer: TrendAnalysis) -> Optional[pd.DataFrame]:
    """Process a single raw data file
    
    Args:
        file_path: Path to raw data file
        analyzer: TrendAnalysis instance
        
    Returns:
        DataFrame with trend analysis results
    """
    try:
        # Load raw data
        data = pd.read_csv(file_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        
        # Extract date from filename (NNE_data_YYYYMMDD.csv)
        date_str = file_path.stem.split('_')[-1]
        
        # Calculate EMAs for trend identification
        data['EMA_short'] = analyzer._calculate_ema(data['Close'], analyzer.ma_short)
        data['EMA_long'] = analyzer._calculate_ema(data['Close'], analyzer.ma_long)
        
        # Identify trends
        trend_data = analyzer.identify_trends(data)
        trend_series = trend_data['Trend']
        strength_series = trend_data['Strength']
        
        # Ensure trend_series and strength_series are pandas Series
        if not isinstance(trend_series, pd.Series) or not isinstance(strength_series, pd.Series):
            raise ValueError("Trend and Strength must be pandas Series")

        # Calculate additional metrics
        strength = analyzer.calculate_trend_strength(data)
        reversals = analyzer.identify_reversal_points(data)
        
        # Add analysis results to data
        results = data.copy()
        results['Trend'] = trend_series
        results['TrendStrength'] = strength_series
        results['IsReversal'] = results['Datetime'].isin([r['time'] for r in reversals])
        
        # Save processed results
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / config.get('data', 'directories', 'analysis') / 'trends'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"Trend_Complete_{date_str}.csv"
        results.to_csv(output_path, index=False)
        
        # Generate and save trend plot
        _generate_trend_plot(results, output_dir / f"Trend_Plot_{date_str}.png")
        
        # Generate and save trend report
        report = analyzer.generate_summary_report(results)
        with open(output_dir / f"Trend_Report_{date_str}.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"Processed {file_path.name}")
        logger.info(f"Saved to: {output_path.absolute()}")
        logger.info(f"Found {len(reversals)} reversal points")
        logger.info(f"Overall trend strength: {strength:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def _generate_trend_plot(data: pd.DataFrame, output_path: Path) -> None:
    """Generate and save trend visualization plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and EMAs
    ax1.plot(data['Datetime'], data['Close'], label='Price', color='black')
    ax1.plot(data['Datetime'], data['EMA_short'], label='EMA Short', 
            alpha=0.7, linestyle='--')
    ax1.plot(data['Datetime'], data['EMA_long'], label='EMA Long', 
            alpha=0.7, linestyle='--')
    
    # Color-code trends
    colors = {'Up': 'green', 'Down': 'red', 'Flat': 'yellow'}
    for trend in colors:
        mask = data['Trend'] == trend
        if mask.any():
            ax1.fill_between(data['Datetime'][mask], data['Close'][mask], 
                           alpha=0.2, color=colors[trend])
    
    # Plot volume
    ax2.bar(data['Datetime'], data['Volume'], color='blue', alpha=0.5)
    
    # Customize plot
    ax1.set_title(f'NNE Price Trends - {output_path.stem}')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_raw_files(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Process multiple raw data files
    
    Args:
        start_date: Optional start date (YYYYMMDD)
        end_date: Optional end date (YYYYMMDD)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize analyzer
        analyzer = TrendAnalysis()
        
        # Get absolute path to raw data directory
        project_root = Path(__file__).resolve().parent.parent.parent
        raw_dir = project_root / config.get('data', 'directories', 'raw')
        
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
            
        logger.info(f"Looking for files in: {raw_dir}")
        
        # Get list of files to process
        files = sorted(raw_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")
            
        # Filter by date range if provided
        if start_date:
            files = [f for f in files if f.stem.split('_')[-1] >= start_date]
        if end_date:
            files = [f for f in files if f.stem.split('_')[-1] <= end_date]
            
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        results = []
        for file_path in files:
            result = process_raw_file(file_path, analyzer)
            if result is not None:
                results.append(result)
                
        # Generate multi-day summary if multiple files processed
        if len(results) > 1:
            summary = analyzer.generate_multi_day_summary(start_date, end_date)
            summary_path = project_root / 'analysis' / 'trends' / 'multi_day_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(summary)
            logger.info(f"Multi-day summary saved to: {summary_path}")
        
        # Print summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total files processed: {len(results)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return []

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
        end_date = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        start_date = end_date = None
        
    process_raw_files(start_date, end_date) 