import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class CountermoveAnalyzer:
    def __init__(self):
        self.countermove_patterns = {
            'small': {'max_price_change': 0.2, 'max_duration': 5, 'typical_volume': 0.8},
            'medium': {'max_price_change': 0.5, 'max_duration': 15, 'typical_volume': 1.0},
            'large': {'max_price_change': 1.0, 'max_duration': 30, 'typical_volume': 1.2},
            'reversal': {'min_price_change': 1.0, 'min_duration': 15, 'min_volume': 1.5}
        }
        self.historical_countermoves = []
        
    def analyze_historical_countermoves(self, data: pd.DataFrame) -> None:
        """Analyze historical price data to identify and categorize countermoves."""
        trend_direction = None
        trend_start_price = None
        trend_start_time = None
        countermove_start = None
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['Close']
            previous_price = data.iloc[i-1]['Close']
            
            # Detect trend
            if trend_direction is None:
                if current_price > previous_price:
                    trend_direction = 'up'
                    trend_start_price = previous_price
                    trend_start_time = data.iloc[i-1]['Datetime']
                elif current_price < previous_price:
                    trend_direction = 'down'
                    trend_start_price = previous_price
                    trend_start_time = data.iloc[i-1]['Datetime']
            
            # Detect countermove
            elif trend_direction == 'up' and current_price < previous_price:
                if countermove_start is None:
                    countermove_start = i-1
            elif trend_direction == 'down' and current_price > previous_price:
                if countermove_start is None:
                    countermove_start = i-1
            
            # Analyze countermove
            if countermove_start is not None:
                countermove_data = data.iloc[countermove_start:i+1]
                countermove_info = self._categorize_countermove(
                    countermove_data,
                    trend_direction,
                    trend_start_price
                )
                
                if countermove_info['type'] != 'ongoing':
                    self.historical_countermoves.append(countermove_info)
                    # Reset if it's a reversal
                    if countermove_info['type'] == 'reversal':
                        trend_direction = 'up' if trend_direction == 'down' else 'down'
                        trend_start_price = current_price
                        trend_start_time = data.iloc[i]['Datetime']
                    countermove_start = None
    
    def _categorize_countermove(self, data: pd.DataFrame, trend_direction: str, trend_start_price: float) -> Dict:
        """Categorize a countermove based on its characteristics."""
        price_change_pct = abs((data.iloc[-1]['Close'] - data.iloc[0]['Close']) / data.iloc[0]['Close'] * 100)
        duration = len(data)
        avg_volume = data['Volume'].mean() / data['Volume'].rolling(20).mean().mean()
        
        # Calculate momentum and volatility
        momentum = data['Close'].diff().mean()
        volatility = data['Close'].pct_change().std()
        
        # Determine countermove type
        if self._is_reversal(price_change_pct, duration, avg_volume):
            countermove_type = 'reversal'
        elif price_change_pct <= self.countermove_patterns['small']['max_price_change']:
            countermove_type = 'small'
        elif price_change_pct <= self.countermove_patterns['medium']['max_price_change']:
            countermove_type = 'medium'
        elif price_change_pct <= self.countermove_patterns['large']['max_price_change']:
            countermove_type = 'large'
        else:
            countermove_type = 'ongoing'
        
        return {
            'type': countermove_type,
            'price_change_pct': price_change_pct,
            'duration': duration,
            'volume_ratio': avg_volume,
            'momentum': momentum,
            'volatility': volatility,
            'start_price': data.iloc[0]['Close'],
            'end_price': data.iloc[-1]['Close'],
            'start_time': data.iloc[0]['Datetime'],
            'end_time': data.iloc[-1]['Datetime'],
            'trend_direction': trend_direction
        }
    
    def _is_reversal(self, price_change_pct: float, duration: int, volume_ratio: float) -> bool:
        """Determine if a countermove is actually a reversal."""
        return (price_change_pct >= self.countermove_patterns['reversal']['min_price_change'] and
                duration >= self.countermove_patterns['reversal']['min_duration'] and
                volume_ratio >= self.countermove_patterns['reversal']['min_volume'])
    
    def analyze_live_countermove(self, current_data: pd.DataFrame, trend_direction: str) -> Dict:
        """Analyze a developing countermove in real-time."""
        countermove_info = self._categorize_countermove(
            current_data,
            trend_direction,
            current_data.iloc[0]['Close']
        )
        
        # Find similar historical patterns
        similar_patterns = self._find_similar_patterns(countermove_info)
        
        # Calculate probability metrics
        reversal_prob = self._calculate_reversal_probability(countermove_info, similar_patterns)
        
        return {
            'current_pattern': countermove_info,
            'similar_patterns_count': len(similar_patterns),
            'reversal_probability': reversal_prob,
            'recommended_action': self._get_recommended_action(reversal_prob, countermove_info)
        }
    
    def _find_similar_patterns(self, current_pattern: Dict) -> List[Dict]:
        """Find historical patterns similar to the current countermove."""
        similar_patterns = []
        
        for historical_pattern in self.historical_countermoves:
            if (abs(historical_pattern['price_change_pct'] - current_pattern['price_change_pct']) < 0.2 and
                abs(historical_pattern['volume_ratio'] - current_pattern['volume_ratio']) < 0.2 and
                historical_pattern['trend_direction'] == current_pattern['trend_direction']):
                similar_patterns.append(historical_pattern)
        
        return similar_patterns
    
    def _calculate_reversal_probability(self, current_pattern: Dict, similar_patterns: List[Dict]) -> float:
        """Calculate the probability of the current countermove becoming a reversal."""
        if not similar_patterns:
            return 0.5  # Default probability when no similar patterns found
        
        # Count how many similar patterns ended up as reversals
        reversal_count = sum(1 for pattern in similar_patterns if pattern['type'] == 'reversal')
        
        # Basic probability
        basic_prob = reversal_count / len(similar_patterns)
        
        # Adjust probability based on current pattern characteristics
        volume_factor = min(current_pattern['volume_ratio'] / 1.5, 1.0)  # Higher volume increases probability
        momentum_factor = abs(current_pattern['momentum']) / 0.01  # Stronger momentum increases probability
        
        adjusted_prob = (basic_prob * 0.6 + volume_factor * 0.2 + momentum_factor * 0.2)
        
        return min(max(adjusted_prob, 0.0), 1.0)  # Ensure probability is between 0 and 1
    
    def _get_recommended_action(self, reversal_prob: float, pattern: Dict) -> str:
        """Get recommended action based on analysis."""
        if reversal_prob > 0.7:
            return "CLOSE_POSITION - High probability of reversal"
        elif reversal_prob > 0.5:
            return "TIGHTEN_STOPS - Medium probability of reversal"
        elif pattern['type'] == 'small':
            return "HOLD - Normal market fluctuation"
        elif pattern['type'] == 'medium':
            return "PARTIAL_TAKE_PROFIT - Consider taking partial profits"
        else:
            return "MONITOR - Continue monitoring pattern development" 