"""
Countermove Analysis Module

Analyzes price movements and trends to identify significant countermoves.
Uses configuration for analysis parameters and output settings.
"""

import pandas as pd
from pathlib import Path

class CountermoveAnalysis:
    """
    Analyzes price movements to identify and analyze countermoves.
    
    Uses configured parameters for:
    - Trend identification
    - Movement analysis
    - Output generation
    - Visualization
    """
    
    def analyze_countermoves(self, trend_data_path: str):
        """
        Simple countermove detection:
        - In downtrend: any uptick is a Rally
        - In uptrend: any downtick is a Pullback
        """
        # Load data
        data = pd.read_csv(trend_data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
        
        # Add countermove column
        data['Countermove'] = 'None'
        
        # Process each row
        for i in range(1, len(data)):
            # Get current trend and price change
            trend = data['Trend'].iloc[i]
            price_change = data['Close'].iloc[i] - data['Close'].iloc[i-1]
            
            # Skip if no trend
            if trend == 'None':
                continue
                
            # Simple countermove rules
            if trend == 'Down' and price_change > 0:  # Any uptick in downtrend
                data.loc[data.index[i], 'Countermove'] = 'Rally'
                
            elif trend == 'Up' and price_change < 0:  # Any downtick in uptrend
                data.loc[data.index[i], 'Countermove'] = 'Pullback'
        
        # Save results
        output_dir = Path('data/countermove_complete')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"countermove_{Path(trend_data_path).name}"
        data.to_csv(output_path)
        
        # Print summary
        print(f"\nCountermove Analysis Complete")
        print(f"Total rows analyzed: {len(data)}")
        print(f"Rallies detected: {len(data[data['Countermove'] == 'Rally'])}")
        print(f"Pullbacks detected: {len(data[data['Countermove'] == 'Pullback'])}")
        print(f"Results saved to: {output_path}")
        
        return data

# Usage example
if __name__ == "__main__":
    analyzer = CountermoveAnalysis()
    trend_file = "data/stock_trend_complete/NNE_trend_data_20241205.csv"
    results = analyzer.analyze_countermoves(trend_file)