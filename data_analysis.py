import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class HistoricalDataAnalysis:
    def __init__(self, data_dir: str = "nne_strategy/data/raw"):
        self.data_dir = Path(data_dir)
        self.all_data = []
        self.daily_stats = {}
        self.overall_stats = {}
        
    def load_all_data(self) -> None:
        """Load all available historical data files."""
        data_files = sorted(self.data_dir.glob("NNE_data_*.csv"))
        print(f"Found {len(data_files)} days of data")
        
        for file in data_files:
            df = pd.read_csv(file)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df['Date'] = df['Datetime'].dt.date
            self.all_data.append(df)
            
            # Calculate daily statistics
            date = df['Date'].iloc[0]
            self.daily_stats[date] = self._calculate_daily_stats(df)
            
        # Combine all data for overall analysis
        self.combined_data = pd.concat(self.all_data)
        self._calculate_overall_stats()
        
    def _calculate_daily_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate key statistics for a single day."""
        return {
            'open': df['Open'].iloc[0],
            'high': df['High'].max(),
            'low': df['Low'].min(),
            'close': df['Close'].iloc[-1],
            'volume': df['Volume'].sum(),
            'price_range': df['High'].max() - df['Low'].min(),
            'price_range_pct': (df['High'].max() - df['Low'].min()) / df['Open'].iloc[0] * 100,
            'trend': 'Up' if df['Close'].iloc[-1] > df['Open'].iloc[0] else 'Down',
            'volatility': df['Close'].pct_change().std() * 100,
            'avg_volume': df['Volume'].mean(),
            'max_up_move': df['Close'].pct_change().max() * 100,
            'max_down_move': df['Close'].pct_change().min() * 100
        }
    
    def _calculate_overall_stats(self) -> None:
        """Calculate overall market statistics."""
        # Price movement patterns
        self.overall_stats['price_patterns'] = {
            'avg_daily_range_pct': np.mean([stats['price_range_pct'] for stats in self.daily_stats.values()]),
            'avg_volatility': np.mean([stats['volatility'] for stats in self.daily_stats.values()]),
            'up_days_pct': sum(1 for stats in self.daily_stats.values() if stats['trend'] == 'Up') / len(self.daily_stats) * 100
        }
        
        # Time-based patterns
        self.overall_stats['time_patterns'] = self._analyze_time_patterns()
        
        # Volume patterns
        self.overall_stats['volume_patterns'] = self._analyze_volume_patterns()
        
        # Movement patterns
        self.overall_stats['movement_patterns'] = self._analyze_movement_patterns()
    
    def _analyze_time_patterns(self) -> Dict:
        """Analyze patterns based on time of day."""
        self.combined_data['Hour'] = self.combined_data['Datetime'].dt.hour
        
        time_patterns = {}
        for hour in range(9, 16):  # Trading hours
            hour_data = self.combined_data[self.combined_data['Hour'] == hour]
            if len(hour_data) > 0:
                time_patterns[hour] = {
                    'avg_price_change': hour_data['Close'].pct_change().mean() * 100,
                    'avg_volume': hour_data['Volume'].mean(),
                    'volatility': hour_data['Close'].pct_change().std() * 100,
                    'success_rate': (hour_data['Close'].pct_change() > 0).mean() * 100
                }
        
        return time_patterns
    
    def _analyze_volume_patterns(self) -> Dict:
        """Analyze volume patterns and their relationship with price movement."""
        df = self.combined_data.copy()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['High_Volume'] = df['Volume'] > df['Volume_MA'] * 1.5
        
        high_volume_moves = df[df['High_Volume']]
        normal_volume_moves = df[~df['High_Volume']]
        
        return {
            'high_volume_success_rate': (high_volume_moves['Close'].pct_change() > 0).mean() * 100,
            'normal_volume_success_rate': (normal_volume_moves['Close'].pct_change() > 0).mean() * 100,
            'high_volume_avg_move': high_volume_moves['Close'].pct_change().mean() * 100,
            'normal_volume_avg_move': normal_volume_moves['Close'].pct_change().mean() * 100
        }
    
    def _analyze_movement_patterns(self) -> Dict:
        """Analyze price movement patterns."""
        df = self.combined_data.copy()
        df['Returns'] = df['Close'].pct_change()
        
        # Define significant moves
        significant_up = df['Returns'] > 0.002  # 0.2% move up
        significant_down = df['Returns'] < -0.002  # 0.2% move down
        
        return {
            'significant_moves_frequency': (significant_up | significant_down).mean() * 100,
            'up_move_frequency': significant_up.mean() * 100,
            'down_move_frequency': significant_down.mean() * 100,
            'avg_up_move': df[significant_up]['Returns'].mean() * 100,
            'avg_down_move': df[significant_down]['Returns'].mean() * 100
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = "NNE Trading Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall Market Behavior
        report += "Overall Market Behavior:\n"
        report += "-" * 30 + "\n"
        report += f"Average Daily Range: {self.overall_stats['price_patterns']['avg_daily_range_pct']:.2f}%\n"
        report += f"Average Volatility: {self.overall_stats['price_patterns']['avg_volatility']:.2f}%\n"
        report += f"Up Days: {self.overall_stats['price_patterns']['up_days_pct']:.1f}%\n\n"
        
        # Time-Based Patterns
        report += "Time-Based Patterns:\n"
        report += "-" * 30 + "\n"
        for hour, stats in self.overall_stats['time_patterns'].items():
            report += f"{hour}:00 - Success Rate: {stats['success_rate']:.1f}%, "
            report += f"Avg Move: {stats['avg_price_change']:.2f}%, "
            report += f"Volume: {stats['avg_volume']:.0f}\n"
        report += "\n"
        
        # Volume Analysis
        report += "Volume Analysis:\n"
        report += "-" * 30 + "\n"
        vol_stats = self.overall_stats['volume_patterns']
        report += f"High Volume Success Rate: {vol_stats['high_volume_success_rate']:.1f}%\n"
        report += f"Normal Volume Success Rate: {vol_stats['normal_volume_success_rate']:.1f}%\n"
        report += f"High Volume Avg Move: {vol_stats['high_volume_avg_move']:.2f}%\n"
        report += f"Normal Volume Avg Move: {vol_stats['normal_volume_avg_move']:.2f}%\n\n"
        
        # Movement Patterns
        report += "Movement Patterns:\n"
        report += "-" * 30 + "\n"
        move_stats = self.overall_stats['movement_patterns']
        report += f"Significant Moves Frequency: {move_stats['significant_moves_frequency']:.1f}%\n"
        report += f"Upward Moves: {move_stats['up_move_frequency']:.1f}% "
        report += f"(Avg: {move_stats['avg_up_move']:.2f}%)\n"
        report += f"Downward Moves: {move_stats['down_move_frequency']:.1f}% "
        report += f"(Avg: {move_stats['avg_down_move']:.2f}%)\n"
        
        return report

def main():
    # Initialize analyzer
    analyzer = HistoricalDataAnalysis()
    
    # Load and analyze data
    print("Loading and analyzing historical data...")
    analyzer.load_all_data()
    
    # Generate and save report
    report = analyzer.generate_report()
    
    # Save report to file
    with open('market_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\nAnalysis complete! Report saved to market_analysis_report.txt")
    print("\nKey Findings:")
    print(report)

if __name__ == "__main__":
    main() 