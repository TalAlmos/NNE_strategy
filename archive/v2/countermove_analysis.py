import pandas as pd
from pathlib import Path

class CountermoveAnalysis:
    def __init__(self):
        """Initialize countermove analysis"""
        self.min_countermove = 0.10  # Minimum price change to consider as countermove
        
    def analyze_countermoves(self, trend_data_path: str):
        """Analyze countermoves in trend data"""
        # Load trend-annotated data
        data = pd.read_csv(trend_data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
        
        # Add countermove column
        data['Countermove'] = 'None'
        
        # Process each row
        for i in range(1, len(data)):
            current_trend = data['Trend'].iloc[i]
            price_change = data['Close'].iloc[i] - data['Close'].iloc[i-1]
            
            # Skip if not in a trend
            if current_trend == 'None':
                continue
                
            # Check for countermove
            if current_trend == 'Up' and price_change < 0:
                # Pullback in uptrend
                if abs(price_change) >= self.min_countermove:
                    data.loc[data.index[i], 'Countermove'] = 'Pullback'
                    
            elif current_trend == 'Down' and price_change > 0:
                # Rally in downtrend
                if abs(price_change) >= self.min_countermove:
                    data.loc[data.index[i], 'Countermove'] = 'Rally'
        
        # Save results
        output_dir = Path('data/countermove_complete')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"countermove_{Path(trend_data_path).name}"
        data.to_csv(output_path)
        
        print(f"\nCountermove analysis completed")
        print(f"Total rows: {len(data)}")
        print(f"Pullbacks identified: {len(data[data['Countermove'] == 'Pullback'])}")
        print(f"Rallies identified: {len(data[data['Countermove'] == 'Rally'])}")
        print(f"Results saved to: {output_path}")
        
        return data

# Usage example
if __name__ == "__main__":
    analyzer = CountermoveAnalysis()
    trend_file = "data/stock_trend_complete/NNE_trend_data_20241205.csv"
    results = analyzer.analyze_countermoves(trend_file)