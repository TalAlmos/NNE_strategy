"""
Trend analysis module for identifying price trends and reversals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import glob

class TrendAnalysis:
    def __init__(self):
        """Initialize trend analyzer with parameters"""
        # Basic trend parameters
        self.min_trend_bars = 2  # Reduced from 3 to be more responsive
        self.smoothing_window = 2  # Reduced from 3 for quicker response
        self.trend_threshold = 0.003  # Reduced from 0.005 for earlier trend detection
        
        # Trend maintenance parameters
        self.max_pullback = 0.004  # Reduced from 0.005 for tighter control
        self.max_pullback_duration = 2  # Keep at 2 bars
        
        # Volume thresholds - relaxed for early morning
        self.volume_threshold = 0.6  # Reduced from 0.7 for early morning trades
        self.volume_weight = 0.25  # Reduced from 0.3 to put less emphasis on volume
        
        # Reversal detection parameters
        self.min_reversal_move = 0.002  # Reduced from 0.003 for quicker detection
        self.min_volume_ratio = 1.1  # Reduced from 1.2 for early morning trades
        self.reversal_confirmation_bars = 2  # Reduced from 3 for faster confirmation
        
        # Momentum calculation
        self.price_window = 8  # Reduced from 10 for faster response
        self.price_momentum_window = 4  # Reduced from 5
        
        # Moving average parameters
        self.fast_ma_period = 4  # Reduced from 5
        self.slow_ma_period = 15  # Reduced from 20
        
        # Early morning specific parameters
        self.early_morning_end_time = "10:00:00"
        self.early_morning_volume_threshold = 0.5  # Even more relaxed volume requirements
        self.early_morning_trend_threshold = 0.002  # More sensitive trend detection
        
        self.ma_short = 8    
        self.ma_long = 21    
        
    def identify_trends(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Identify trends in the data"""
        # Example implementation
        trend_series = pd.Series(index=data.index, dtype='object')
        strength_series = pd.Series(index=data.index, dtype='float')

        # Logic to populate trend_series and strength_series
        # ...

        return {'Trend': trend_series, 'Strength': strength_series}
        
    def identify_reversal_points(self, data: pd.DataFrame) -> List[Dict]:
        """Identify trend reversal points"""
        reversals = []
        window_size = 5  # Look at 5-minute windows
        
        for i in range(window_size, len(data)-window_size):
            if self._is_reversal_point(data, i, window_size):
                reversals.append({
                    'time': data['Datetime'].iloc[i],
                    'price': data['Close'].iloc[i],
                    'type': 'High' if self._is_peak(data, i, window_size) else 'Low'
                })
                
        return reversals
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        price_changes = data['Close'].pct_change()
        return abs(price_changes.mean()) / price_changes.std()
    
    def _determine_trend(self, data: pd.DataFrame, idx: int) -> str:
        """Determine trend direction at a specific point"""
        if idx < 20:  # Need enough data for trend determination
            return 'None'  # Start with None for initial points
        
        # Use a longer window to smooth out countermoves
        window = 20
        start_idx = max(0, idx - window)
        window_data = data.iloc[start_idx:idx+1]
        
        # Calculate price change over the window
        price_change = (window_data['Close'].iloc[-1] - window_data['Close'].iloc[0]) / window_data['Close'].iloc[0]
        
        # Use larger thresholds to ignore small movements
        if abs(price_change) < 0.003:  # 0.3% minimum change for trend
            return 'Flat'
        
        return 'Up' if price_change > 0 else 'Down'
    
    def _calculate_trend_strength(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate trend strength at a specific point"""
        window = 5
        if idx < window:
            return 0.0
            
        prices = data['Close'].iloc[idx-window:idx+1]
        return abs(prices.pct_change().mean())
    
    def _is_reversal_point(self, data: pd.DataFrame, idx: int, window: int) -> bool:
        """Check if point is a trend reversal"""
        return self._is_peak(data, idx, window) or self._is_trough(data, idx, window)
    
    def _is_peak(self, data: pd.DataFrame, idx: int, window: int) -> bool:
        """Check if point is a peak"""
        before = data['Close'].iloc[idx-window:idx].max()
        after = data['Close'].iloc[idx+1:idx+window+1].max()
        current = data['Close'].iloc[idx]
        return current > before and current > after
    
    def _is_trough(self, data: pd.DataFrame, idx: int, window: int) -> bool:
        """Check if point is a trough"""
        before = data['Close'].iloc[idx-window:idx].min()
        after = data['Close'].iloc[idx+1:idx+window+1].min()
        current = data['Close'].iloc[idx]
        return current < before and current < after
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def _is_trend_starting(self, data: pd.DataFrame, i: int, is_early_morning: bool = False) -> bool:
        """Check if a new trend is starting"""
        if i < self.min_trend_bars:
            return False
            
        price = data['Close'].iloc[i]
        prev_prices = data['Close'].iloc[i-self.min_trend_bars:i]
        
        # More sensitive price direction check in early morning
        threshold = self.early_morning_trend_threshold if is_early_morning else self.trend_threshold
        price_change = (price - prev_prices.mean()) / prev_prices.mean()
        
        # Check price direction with adjusted threshold
        is_higher = price_change > threshold
        is_lower = price_change < -threshold
        
        # Check moving average alignment with early morning consideration
        ma_aligned = False
        if not pd.isna(data['SMA5'].iloc[i]) and not pd.isna(data['SMA20'].iloc[i]):
            ma_diff = data['SMA5'].iloc[i] - data['SMA20'].iloc[i]
            # More lenient MA alignment requirement in early morning
            if is_early_morning:
                ma_aligned = (ma_diff > -threshold and is_higher) or (ma_diff < threshold and is_lower)
            else:
                ma_aligned = (ma_diff > 0 and is_higher) or (ma_diff < 0 and is_lower)
        
        return (is_higher or is_lower) and (ma_aligned or is_early_morning)
    
    def _confirm_reversal(self, data: pd.DataFrame, i: int, direction: str, is_early_morning: bool = False) -> bool:
        """Confirm trend reversal with additional checks"""
        if i < self.reversal_confirmation_bars:
            return False
            
        # Get recent data
        recent_data = data.iloc[i-self.reversal_confirmation_bars:i+1]
        
        # Check if it's a countermove
        if self.is_countermove(recent_data):
            return False
            
        # Check reversal signal
        reversal = self.detect_reversal(recent_data)
        if not reversal['detected'] or reversal['direction'] != direction:
            return False
            
        # Additional volume confirmation
        volume_increasing = recent_data['Volume'].iloc[-1] > recent_data['Volume'].iloc[:-1].mean()
        
        return volume_increasing
    
    def _start_new_trend(self, data: pd.DataFrame, idx: int, direction: str = None) -> Dict:
        """Initialize a new trend"""
        if direction is None:
            # Determine direction from price action
            price_change = data['Close'].iloc[idx] - data['Close'].iloc[idx-1]
            direction = 'Up' if price_change > 0 else 'Down'
            
        return {
            'direction': direction,
            'start_idx': idx,
            'start_price': data['Close'].iloc[idx],
            'high': data['Close'].iloc[idx],
            'low': data['Close'].iloc[idx],
            'strength': self.calculate_trend_strength(data.iloc[max(0, idx-5):idx+1])
        }
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required)
        
    def _analyze_window(self, prices: np.ndarray) -> tuple:
        """Analyze price window for trend"""
        # Calculate linear regression
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Determine trend direction and strength
        if abs(slope) < self.trend_threshold:
            return 'None', 0.0
            
        direction = 'Up' if slope > 0 else 'Down'
        strength = min(1.0, abs(slope) / self.trend_threshold)
        
        return direction, strength
        
    def _get_trend_direction(self, prices: np.ndarray) -> str:
        """Get trend direction for array of prices"""
        if len(prices) < 2:
            return 'None'
            
        slope = np.polyfit(range(len(prices)), prices, 1)[0]
        
        if abs(slope) < self.trend_threshold:
            return 'None'
            
        return 'Up' if slope > 0 else 'Down' 
        
    def _calculate_volume_strength(self, volumes: np.ndarray) -> float:
        """Calculate trend strength based on volume"""
        if len(volumes) < 2:
            return 0.0
            
        # Calculate volume trend
        vol_slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
        return min(1.0, abs(vol_slope) / np.mean(volumes))
        
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        if len(prices) < self.price_momentum_window:
            return 0.0
            
        # Use rate of change
        roc = (prices[-1] - prices[-self.price_momentum_window]) / prices[-self.price_momentum_window]
        return min(1.0, abs(roc) / self.trend_threshold) 
        
    def generate_summary_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate a detailed trend analysis report including trend durations and price changes
        """
        if not self._validate_data(data):
            return {}
        
        trends = self.identify_trends(data)
        
        # Initialize report sections
        report = {
            'period_start': data.iloc[0]['Datetime'],
            'period_end': data.iloc[-1]['Datetime'],
            'trends': [],
            'countermoves': []
        }
        
        # Track trend changes
        current_trend = None
        trend_start_idx = 0
        trend_count = 0
        
        for i in range(len(trends)):
            if trends.iloc[i]['Trend'] != current_trend and trends.iloc[i]['Trend'] != 'None':
                if current_trend is not None:
                    # Calculate trend metrics
                    duration_mins = (data.iloc[i]['Datetime'] - 
                                   data.iloc[trend_start_idx]['Datetime']).total_seconds() / 60
                    price_change = data.iloc[i]['Close'] - data.iloc[trend_start_idx]['Close']
                    pct_change = (price_change / data.iloc[trend_start_idx]['Close']) * 100
                    
                    trend_info = {
                        'direction': current_trend,
                        'duration': int(duration_mins),
                        'price_change': round(price_change, 2),
                        'percent_change': round(pct_change, 1)
                    }
                    report['trends'].append(trend_info)
                    trend_count += 1
                
                current_trend = trends.iloc[i]['Trend']
                trend_start_idx = i
        
        # Format the report as text
        report_text = "Analysis Report for NNE\n"
        report_text += "=" * 50 + "\n\n"
        
        report_text += f"Analysis Period: {report['period_start']} to {report['period_end']}\n"
        report_text += f"Total Trends Identified: {trend_count}\n"
        report_text += f"Total Countermoves: {len(report['countermoves'])}\n\n"
        
        report_text += "Trend Analysis:\n"
        report_text += "-" * 30 + "\n\n"
        
        for trend in report['trends']:
            report_text += f"{trend['direction']} Trend:\n"
            report_text += f"  Duration: {trend['duration']} minutes\n"
            report_text += f"  Price Change: ${trend['price_change']} ({trend['percent_change']}%)\n\n"
        
        return report_text
    
    def _format_price(self, price: float) -> str:
        """Format price with appropriate precision"""
        return f"${price:.2f}"
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key support and resistance levels"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Simple method using recent highs/lows
        support_levels = sorted(set(np.percentile(lows[-20:], [20, 40, 60])))
        resistance_levels = sorted(set(np.percentile(highs[-20:], [40, 60, 80])))
        
        return {
            'support': [float(level) for level in support_levels],
            'resistance': [float(level) for level in resistance_levels]
        }
    
    def generate_and_save_report(self, data: pd.DataFrame, output_path: str = "analysis_results/NNE_analysis.txt") -> None:
        """
        Generate trend analysis report and save it to a file
        
        Args:
            data: DataFrame with price data
            output_path: Path where to save the report (default: analysis_results/NNE_analysis.txt)
        """
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate report
        report_text = self.generate_summary_report(data)
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")
    
    def _get_significant_trends(self, trends: List[Dict]) -> List[Dict]:
        """Identify the most significant trends based on duration and price change"""
        significant = []
        for trend in trends:
            # Consider a trend significant if:
            # - Duration >= 10 minutes OR
            # - Price change >= 1% OR
            # - Average strength >= 0.8
            if (trend['duration'] >= 10 or 
                abs(trend['percent_change']) >= 1.0 or 
                trend['avg_strength'] >= 0.8):
                significant.append(trend)
        return significant
    
    def generate_multi_day_summary(self, start_date: str = None, end_date: str = None, data_dir: str = "data/analysis/trends/") -> str:
        """Generate a summary report with consolidated trends, filtering out noise"""
        pattern = os.path.join(data_dir, "trend_analysis_*.csv")
        files = sorted(glob.glob(pattern))
        
        if start_date:
            files = [f for f in files if f[-12:-4] >= start_date]
        if end_date:
            files = [f for f in files if f[-12:-4] <= end_date]
        
        report = "Multi-Day Trend Analysis Summary\n"
        report += "=" * 50 + "\n\n"
        
        for file in files:
            date = file[-12:-4]
            df = pd.read_csv(file)
            
            # Calculate daily statistics
            day_open = df.iloc[0]['Open']
            day_close = df.iloc[-1]['Close']
            day_high = df['High'].max()
            day_low = df['Low'].min()
            price_change = day_close - day_open
            pct_change = (price_change / day_open) * 100
            
            # Consolidate consecutive trends with improved filtering
            consolidated_trends = []
            current_trend = None
            trend_start_idx = 0
            trend_start_time = None
            min_trend_duration = 1  # Minimum trend duration in minutes
            
            for i in range(len(df)):
                row = df.iloc[i]
                if row['Trend'] != current_trend:
                    if current_trend is not None and current_trend != 'None':
                        # Calculate consolidated trend metrics
                        end_time = df.iloc[i-1]['Datetime']
                        duration = pd.Timestamp(end_time) - pd.Timestamp(trend_start_time)
                        duration_mins = duration.total_seconds() / 60
                        
                        # Only include trends longer than minimum duration
                        if duration_mins >= min_trend_duration:
                            price_change = df.iloc[i-1]['Close'] - df.iloc[trend_start_idx]['Open']
                            pct_change = (price_change / df.iloc[trend_start_idx]['Open']) * 100
                            avg_strength = df.iloc[trend_start_idx:i]['TrendStrength'].mean()
                            
                            consolidated_trends.append({
                                'direction': current_trend,
                                'start_time': trend_start_time,
                                'end_time': end_time,
                                'duration': int(duration_mins),
                                'price_change': round(price_change, 2),
                                'percent_change': round(pct_change, 1),
                                'avg_strength': round(avg_strength, 2)
                            })
                    
                    if row['Trend'] != 'None' and row['TrendStrength'] > 0.1:  # Filter out weak trends
                        current_trend = row['Trend']
                        trend_start_idx = i
                        trend_start_time = row['Datetime']
                    else:
                        current_trend = None
            
            # Format daily summary with additional statistics
            report += f"Date: {date[:4]}-{date[4:6]}-{date[6:]}\n"
            report += "-" * 30 + "\n"
            report += f"Open: ${day_open:.2f}\n"
            report += f"Close: ${day_close:.2f}\n"
            report += f"High: ${day_high:.2f}\n"
            report += f"Low: ${day_low:.2f}\n"
            report += f"Change: ${price_change:.2f} ({pct_change:.1f}%)\n\n"
            
            # Add trend summary statistics
            up_trends = [t for t in consolidated_trends if t['direction'] == 'Up']
            down_trends = [t for t in consolidated_trends if t['direction'] == 'Down']
            
            report += "Daily Summary:\n"
            report += f"Total Trends: {len(consolidated_trends)}\n"
            report += f"Uptrends: {len(up_trends)} | Downtrends: {len(down_trends)}\n"
            report += f"Longest Trend: {max([t['duration'] for t in consolidated_trends], default=0)} minutes\n\n"
            
            # Add major price moves first
            major_moves = [t for t in consolidated_trends if abs(t['percent_change']) >= 1.0]
            if major_moves:
                report += "Major Price Moves:\n"
                for move in major_moves:
                    report += f"- {move['direction']}: {move['percent_change']}% in {move['duration']}min "
                    report += f"({move['start_time'].split()[1]} - {move['end_time'].split()[1]})\n"
                report += "\n"
            
            # Add key reversal points
            reversals = []
            for i in range(1, len(consolidated_trends)):
                prev = consolidated_trends[i-1]
                curr = consolidated_trends[i]
                if prev['direction'] != curr['direction'] and (abs(prev['percent_change']) >= 0.5 or abs(curr['percent_change']) >= 0.5):
                    reversals.append({
                        'time': curr['start_time'],
                        'price': curr['price_change'],
                        'from_trend': prev['direction'],
                        'to_trend': curr['direction'],
                        'strength': curr['avg_strength']
                    })
            
            if reversals:
                report += "Key Reversals:\n"
                for rev in reversals:
                    report += f"- {rev['time'].split()[1]}: {rev['from_trend']} -> {rev['to_trend']} "
                    report += f"at ${rev['price']:.2f} (strength: {rev['strength']})\n"
                report += "\n"
            
            # Add strongest trends
            strong_trends = [t for t in consolidated_trends if t['avg_strength'] >= 0.8]
            if strong_trends:
                report += "Strong Trends:\n"
                for trend in strong_trends:
                    report += f"- {trend['direction']}: {trend['duration']}min with {trend['avg_strength']} strength "
                    report += f"({trend['percent_change']}% move)\n"
                report += "\n"
            
            report += "=" * 50 + "\n\n"
        
        return report
    
    def _calculate_initial_strength(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate initial trend strength"""
        return self.calculate_trend_strength(data.iloc[max(0, idx-5):idx+1])
    
    def is_countermove(self, data: pd.DataFrame, volume_threshold: float = None) -> bool:
        """Identify if current down/up move is a countermove rather than true reversal"""
        if len(data) < 3:  # Reduced from 4
            return False
            
        # Get recent price action
        recent_data = data.tail(3)  # Reduced from 4
        initial_move = recent_data['Close'].iloc[1] - recent_data['Close'].iloc[0]
        subsequent_move = recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[1]
        
        # Volume analysis
        initial_volume = recent_data['Volume'].iloc[1]
        avg_subsequent_volume = recent_data['Volume'].iloc[2:].mean()
        
        # Use provided volume threshold or default
        vol_threshold = volume_threshold if volume_threshold is not None else self.volume_threshold
        
        if initial_move > 0:  # Upward initial move
            return (subsequent_move < 0 and  # Downward countermove
                    abs(subsequent_move) < abs(initial_move) * 0.8 and  # Less strength
                    avg_subsequent_volume < initial_volume * vol_threshold)  # Lower volume
        else:  # Downward initial move
            return (subsequent_move > 0 and  # Upward countermove
                    abs(subsequent_move) < abs(initial_move) * 0.8 and  # Less strength
                    avg_subsequent_volume < initial_volume * vol_threshold)  # Lower volume
    
    def detect_reversal(self, data: pd.DataFrame, is_early_morning: bool = False) -> Dict:
        """Detect if current price action indicates a reversal"""
        price_changes = data['Close'].pct_change()
        volume_ratio = data['Volume'] / data['Volume'].rolling(5).mean()
        
        # Adjust thresholds for early morning
        min_move = self.early_morning_trend_threshold if is_early_morning else self.min_reversal_move
        vol_ratio = self.min_volume_ratio * (0.9 if is_early_morning else 1.0)
        
        # Look for initial surge with adjusted conditions
        surge_detected = (
            abs(price_changes.iloc[-1]) > min_move and  # Minimum move
            (volume_ratio.iloc[-1] > vol_ratio or is_early_morning) and  # Volume confirmation
            # Check if move is continuing the surge direction
            (price_changes.iloc[-1] > 0 and data['Close'].iloc[-1] > data['Close'].iloc[-2]) or
            (price_changes.iloc[-1] < 0 and data['Close'].iloc[-1] < data['Close'].iloc[-2])
        )
        
        if surge_detected:
            direction = 'Up' if price_changes.iloc[-1] > 0 else 'Down'
            return {
                'detected': True,
                'direction': direction,
                'strength': abs(price_changes.iloc[-1]),
                'volume_ratio': volume_ratio.iloc[-1],
                'price': data['Close'].iloc[-1]
            }
        
        return {'detected': False}