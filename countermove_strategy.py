from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional
from nne_strategy.counter_move_stats import CounterMoveStats

class CountermoveStrategy:
    def __init__(self, entry_style='conservative'):
        """Initialize strategy parameters"""
        # Entry style
        self.entry_style = entry_style  # 'aggressive' or 'conservative'
        
        # Original parameters
        self.min_trend_duration = 15     
        self.min_trend_strength = 0.6    
        self.min_volume_ratio = 1.2      
        
        # Updated parameters to match design
        self.min_trend_change = 0.40     # $0.40 minimum trend move
        self.expected_countermove_ratio = 0.50  # 50% expected retracement
        
        # Risk parameters from design
        self.max_position_size = 0.20    # 20% max position
        self.daily_loss_limit = 0.02     # 2% daily loss limit
        self.trailing_stop = 0.005       # 0.5% trailing stop
        
        # Load historical statistics
        self.stats = CounterMoveStats()
    
    def calculate_position_size(self, capital: float, entry_price: float, risk_per_trade: float = 0.01) -> int:
        """Calculate position size based on risk parameters
        
        Args:
            capital: Available trading capital
            entry_price: Planned entry price
            risk_per_trade: Risk per trade (default 1%)
            
        Returns:
            int: Number of shares to trade
        """
        risk_amount = capital * risk_per_trade
        position_value = capital * self.max_position_size
        max_shares = int(position_value / entry_price)
        
        # Calculate shares based on stop loss
        stop_distance = entry_price * self.trailing_stop
        risk_based_shares = int(risk_amount / stop_distance)
        
        return min(risk_based_shares, max_shares)
    
    def evaluate_trend(self, trend_data: Dict) -> Dict:
        """Evaluate trend according to design specifications
        
        Args:
            trend_data: Current trend information
            
        Returns:
            Dict with trend evaluation results
        """
        duration = self._calculate_duration(trend_data)
        price_change = abs(trend_data['end_price'] - trend_data['start_price'])
        
        return {
            'valid': duration >= self.min_trend_duration and price_change >= self.min_trend_change,
            'metrics': {
                'duration': duration,
                'price_change': price_change,
                'expected_countermove': price_change * self.expected_countermove_ratio
            }
        }
    
    def analyze_trend(self, trend_data: Dict, price_data: pd.DataFrame) -> Dict:
        """
        Analyze trend characteristics for trading suitability
        
        Args:
            trend_data: Current trend information
            price_data: Recent price and volume data
        """
        # Calculate trend metrics
        duration = self._calculate_duration(trend_data)
        strength = self._calculate_trend_strength(price_data)
        volume_confirmed = self._check_volume_confirmation(price_data)
        
        return {
            'suitable': (duration >= self.min_trend_duration and 
                       strength >= self.min_trend_strength and 
                       volume_confirmed),
            'metrics': {
                'duration': duration,
                'strength': strength,
                'volume_confirmed': volume_confirmed,
                'trend_type': trend_data['direction']
            }
        }
    
    def analyze_countermove(self, 
                          trend_data: Dict,
                          current_data: pd.DataFrame,
                          lookback_window: pd.DataFrame) -> Dict:
        """
        Analyze countermove pattern for trading opportunity
        
        Args:
            trend_data: Current trend information
            current_data: Current price bar
            lookback_window: Recent price data for analysis
        """
        # Calculate countermove characteristics
        move_size = self._calculate_countermove_size(
            trend_data['direction'],
            lookback_window,
            current_data
        )
        
        volume_pattern = self._analyze_volume_pattern(lookback_window)
        momentum = self._calculate_momentum(lookback_window)
        
        # Determine if countermove is valid for trading
        valid_size = (self.min_countermove_size <= move_size <= self.max_countermove_size)
        valid_volume = volume_pattern['decreasing'] and volume_pattern['ratio'] <= self.countermove_volume_factor
        
        return {
            'valid': valid_size and valid_volume,
            'metrics': {
                'move_size': move_size,
                'volume_pattern': volume_pattern,
                'momentum': momentum
            }
        }
    
    def calculate_trade_levels(self, 
                             trend_data: Dict,
                             countermove_data: Dict,
                             current_price: float) -> Dict:
        """
        Calculate entry, target, and risk levels
        
        Args:
            trend_data: Current trend information
            countermove_data: Countermove analysis results
            current_price: Current market price
        """
        trend_move = abs(trend_data['end_price'] - trend_data['start_price'])
        countermove_size = countermove_data['metrics']['move_size']
        
        # Calculate completion targets
        min_target = trend_move * self.min_completion_ratio
        max_target = trend_move * self.max_completion_ratio
        
        if trend_data['direction'] == 'UpTrend':
            target_price = current_price + max_target
            min_target_price = current_price + min_target
        else:
            target_price = current_price - max_target
            min_target_price = current_price - min_target
        
        return {
            'entry_price': current_price,
            'target_price': target_price,
            'min_target': min_target_price,
            'trend_size': trend_move,
            'countermove_size': countermove_size
        }
    
    def _calculate_duration(self, trend_data: Dict) -> float:
        """Calculate trend duration in minutes"""
        start_time = pd.to_datetime(trend_data['start_time'])
        end_time = pd.to_datetime(trend_data['end_time'])
        return (end_time - start_time).total_seconds() / 60
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength score"""
        closes = price_data['Close'].values
        highs = price_data['High'].values
        lows = price_data['Low'].values
        
        # Calculate directional movement
        up_moves = np.sum(closes[1:] > closes[:-1])
        down_moves = np.sum(closes[1:] < closes[:-1])
        
        # Calculate price range usage
        range_usage = (closes[-1] - closes[0]) / (max(highs) - min(lows))
        
        return (max(up_moves, down_moves) / len(closes)) * abs(range_usage)
    
    def _check_volume_confirmation(self, price_data: pd.DataFrame) -> bool:
        """Check if volume confirms the trend"""
        volume = price_data['Volume'].values
        avg_volume = np.mean(volume)
        recent_volume = np.mean(volume[-3:])  # Last 3 bars
        
        return recent_volume >= (avg_volume * self.min_volume_ratio)
    
    def _calculate_countermove_size(self,
                                  trend_type: str,
                                  window: pd.DataFrame,
                                  current: pd.DataFrame) -> float:
        """Calculate countermove size relative to trend"""
        if trend_type == 'UpTrend':
            recent_high = window['High'].max()
            return (recent_high - current['Close'].iloc[0]) / recent_high
        else:
            recent_low = window['Low'].min()
            return (current['Close'].iloc[0] - recent_low) / recent_low
    
    def _analyze_volume_pattern(self, window: pd.DataFrame) -> Dict:
        """Analyze volume pattern during countermove"""
        volume = window['Volume'].values
        decreasing = np.all(volume[1:] <= volume[:-1])
        volume_ratio = volume[-1] / volume[0]
        
        return {
            'decreasing': decreasing,
            'ratio': volume_ratio
        }
    
    def _calculate_momentum(self, window: pd.DataFrame) -> float:
        """Calculate price momentum"""
        closes = window['Close'].values
        return (closes[-1] - closes[0]) / closes[0]