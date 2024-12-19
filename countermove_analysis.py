import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class CountermoveAnalysis:
    def __init__(self):
        """Initialize countermove analysis with optimized parameters"""
        # Core parameters
        self.min_countermove = 0.05  # 5% minimum move
        self.max_countermove = 0.30  # 30% maximum move
        self.min_trend_strength = 0.65
        
        # Time filters
        self.min_trend_duration = 3
        self.max_countermove_duration = 5
        
        # Performance thresholds
        self.success_threshold = 0.70
        self.failure_threshold = 1.50
    
    def analyze_countermoves(self, trend_data_path: str) -> pd.DataFrame:
        """
        Analyze countermoves in trend data with detailed classification
        """
        # Load and prepare data
        data = pd.read_csv(trend_data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
        
        # Initialize countermove columns
        data['Countermove'] = 'None'
        data['CountermoveStrength'] = 0.0
        data['CountermoveSuccess'] = False
        
        # Track active countermoves
        active_countermove = {
            'type': None,
            'start_idx': None,
            'start_price': None,
            'max_deviation': 0.0
        }
        
        # Process each bar
        for i in range(1, len(data)):
            current_trend = data['Trend'].iloc[i]
            current_price = data['Close'].iloc[i]
            prev_price = data['Close'].iloc[i-1]
            price_change = current_price - prev_price
            trend_strength = data['TrendStrength'].iloc[i]
            
            # Skip if trend is too weak or no trend
            if current_trend == 'None' or trend_strength < self.min_trend_strength:
                continue
            
            # Check for countermove conditions
            if self._is_countermove(current_trend, price_change, active_countermove):
                if active_countermove['type'] is None:
                    # Start new countermove
                    active_countermove = self._start_countermove(
                        current_trend, i, current_price, price_change
                    )
                    data.loc[data.index[i], 'Countermove'] = active_countermove['type']
                else:
                    # Continue existing countermove
                    deviation = abs(current_price - active_countermove['start_price'])
                    active_countermove['max_deviation'] = max(
                        active_countermove['max_deviation'], 
                        deviation
                    )
                    
                    # Check if countermove exceeds maximum
                    if deviation > self.max_countermove:
                        self._reset_countermove(active_countermove)
                    else:
                        data.loc[data.index[i], 'Countermove'] = active_countermove['type']
                        data.loc[data.index[i], 'CountermoveStrength'] = deviation / self.max_countermove
            
            else:
                # Check if countermove completed successfully
                if active_countermove['type'] is not None:
                    success = self._evaluate_countermove_success(
                        current_price, 
                        active_countermove
                    )
                    if success:
                        data.loc[data.index[i], 'CountermoveSuccess'] = True
                    self._reset_countermove(active_countermove)
        
        # Save results
        self._save_results(data, trend_data_path)
        self._print_summary(data)
        
        return data
    
    def _is_countermove(self, trend: str, price_change: float, 
                       active: Dict) -> bool:
        """Determine if price action represents a countermove"""
        if active['type'] is not None:
            # Already in countermove, check if continuing
            return (
                (active['type'] == 'Pullback' and price_change < 0) or
                (active['type'] == 'Rally' and price_change > 0)
            )
        else:
            # Check for new countermove
            return (
                (trend == 'Up' and price_change < -self.min_countermove) or
                (trend == 'Down' and price_change > self.min_countermove)
            )
    
    def _start_countermove(self, trend: str, idx: int, 
                          price: float, change: float) -> Dict:
        """Initialize a new countermove"""
        return {
            'type': 'Pullback' if trend == 'Up' else 'Rally',
            'start_idx': idx,
            'start_price': price,
            'max_deviation': abs(change)
        }
    
    def _evaluate_countermove_success(self, current_price: float, 
                                    countermove: Dict) -> bool:
        """Determine if countermove completed successfully"""
        if countermove['type'] == 'Pullback':
            return current_price > countermove['start_price'] * self.success_threshold
        else:
            return current_price < countermove['start_price'] * self.success_threshold
    
    def _reset_countermove(self, countermove: Dict) -> None:
        """Reset countermove tracking"""
        countermove['type'] = None
        countermove['start_idx'] = None
        countermove['start_price'] = None
        countermove['max_deviation'] = 0.0
    
    def _save_results(self, data: pd.DataFrame, input_path: str) -> None:
        """Save analysis results"""
        output_dir = Path('data/countermove_complete')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"countermove_{Path(input_path).name}"
        data.to_csv(output_path)
    
    def _print_summary(self, data: pd.DataFrame) -> None:
        """Print analysis summary"""
        pullbacks = len(data[data['Countermove'] == 'Pullback'])
        rallies = len(data[data['Countermove'] == 'Rally'])
        successful = len(data[data['CountermoveSuccess']])
        
        print("\nCountermove Analysis Summary")
        print("=" * 30)
        print(f"Total Bars Analyzed: {len(data)}")
        print(f"Pullbacks Identified: {pullbacks}")
        print(f"Rallies Identified: {rallies}")
        print(f"Successful Countermoves: {successful}")
        print(f"Success Rate: {(successful/(pullbacks+rallies)*100):.1f}%")

# Usage example
if __name__ == "__main__":
    analyzer = CountermoveAnalysis()
    trend_file = "nne_strategy/data/analysis/trends/trend_analysis_20241205.csv"
    results = analyzer.analyze_countermoves(trend_file)