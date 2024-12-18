"""
Counter Move Statistics Module

Analyzes counter-movements in price trends and calculates relevant statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class CountermoveStats:
    """Analyzes countermove patterns in stock data"""
    
    def analyze_countermove_patterns(self, countermove_data_path: str, 
                                   min_duration: int = 2,
                                   min_price_change_pct: float = 0.2) -> pd.DataFrame:
        """
        Analyzes countermove patterns from processed data
        
        Args:
            countermove_data_path: Path to countermove CSV file
            min_duration: Minimum number of bars for a valid pattern
            min_price_change_pct: Minimum price change % for a valid pattern
        """
        # Load countermove data
        data = pd.read_csv(countermove_data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
        
        # Get symbol and date from filename
        file_name = Path(countermove_data_path).name
        symbol = file_name.split('_')[1]
        date = file_name.split('_')[3].split('.')[0]
        
        # Initialize results storage
        patterns = []
        
        # Find all countermove sequences
        i = 0
        while i < len(data):
            if data['Countermove'].iloc[i] != 'None':
                # Found start of pattern
                pattern_type = data['Countermove'].iloc[i]
                start_idx = i
                
                # Find end of consecutive same-type countermoves
                j = i + 1
                while j < len(data) and data['Countermove'].iloc[j] == pattern_type:
                    j += 1
                
                # Only analyze patterns meeting minimum criteria
                if j - start_idx >= min_duration:
                    pattern_stats = self._analyze_single_pattern(
                        data,
                        start_idx,
                        j - 1,
                        pattern_type
                    )
                    
                    # Add pattern if it meets minimum price change
                    if abs(pattern_stats['price_change_pct']) >= min_price_change_pct:
                        patterns.append(pattern_stats)
                
                i = j  # Move to end of pattern
            else:
                i += 1
        
        # Create DataFrame from patterns
        results_df = pd.DataFrame(patterns)
        
        # Calculate probabilities
        prob_stats = self._calculate_probabilities(results_df)
        
        # Save results
        output_dir = Path('data/countermove_dataset')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"countermove_stats_{symbol}_{date}.csv"
        results_df.to_csv(output_path)
        
        # Print summary
        print("\nCountermove Pattern Analysis")
        print("=" * 50)
        print(f"\nTotal patterns found: {len(results_df)}")
        
        if len(results_df) > 0:
            print("\nPattern type distribution:")
            print(results_df['pattern_type'].value_counts())
            print("\nPattern strength distribution:")
            print(results_df['pattern_strength'].value_counts())
            
            print("\nProbability Analysis:")
            print("=" * 30)
            
            # Print probability stats for each strength level
            for strength in ['Strong', 'Moderate', 'Weak']:
                if strength in prob_stats:
                    stats = prob_stats[strength]
                    print(f"\n{strength} Patterns:")
                    print(f"Total count: {stats['total']}")
                    print(f"Resume probability: {stats['resume_prob']:.1%}")
                    print(f"Reversal probability: {stats['reversal_prob']:.1%}")
                    print(f"Average recovery bars: {stats['avg_recovery_bars']:.1f}")
                    print(f"Average price change: {stats['avg_price_change_pct']:.2f}%")
                    print(f"Average volume change: {stats['avg_volume_change_pct']:.2f}%")
            
            print("\nPattern Type Analysis:")
            print("=" * 30)
            for pattern in ['Rally', 'Pullback']:
                type_stats = prob_stats.get(f'{pattern}_stats', {})
                if type_stats:
                    print(f"\n{pattern} Patterns:")
                    print(f"Resume probability: {type_stats['resume_prob']:.1%}")
                    print(f"Average duration: {type_stats['avg_duration']:.1f} bars")
                    print(f"Average price change: {type_stats['avg_price_change_pct']:.2f}%")
        
        print(f"\nResults saved to: {output_path}")
        
        return results_df
    
    def _analyze_single_pattern(self, data: pd.DataFrame, 
                              start_idx: int, end_idx: int, 
                              pattern_type: str) -> Dict:
        """Analyzes a single countermove pattern"""
        pattern_data = data.iloc[start_idx:end_idx+1]
        
        # Calculate price stats
        start_price = pattern_data['Close'].iloc[0]
        end_price = pattern_data['Close'].iloc[-1]
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100
        
        # Calculate pattern characteristics
        duration = len(pattern_data)
        max_price = pattern_data['High'].max()
        min_price = pattern_data['Low'].min()
        price_range = max_price - min_price
        price_range_pct = (price_range / start_price) * 100
        
        # Calculate risk metrics
        if pattern_type == 'Pullback':
            # For pullbacks in uptrend
            risk = end_price - min_price  # Risk to lowest point
            potential_reward = max_price - end_price  # Reward to highest point
        else:
            # For rallies in downtrend
            risk = max_price - end_price  # Risk to highest point
            potential_reward = end_price - min_price  # Reward to lowest point
        
        risk_pct = (risk / end_price) * 100
        reward_pct = (potential_reward / end_price) * 100
        risk_reward_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        # Calculate stop loss and target levels
        stop_loss = min_price if pattern_type == 'Pullback' else max_price
        target_1r = end_price + (risk if pattern_type == 'Pullback' else -risk)  # 1:1 R/R
        target_2r = end_price + (2 * risk if pattern_type == 'Pullback' else -2 * risk)  # 2:1 R/R
        
        # Volume analysis
        avg_volume = pattern_data['Volume'].mean()
        volume_change_pct = ((pattern_data['Volume'].iloc[-1] / pattern_data['Volume'].iloc[0]) - 1) * 100
        
        # Classify pattern strength
        strength = self._classify_pattern_strength(
            duration, price_change_pct, price_range_pct, volume_change_pct
        )
        
        # Look ahead for outcome and targets
        look_ahead = min(10, len(data) - end_idx)
        future_data = data.iloc[end_idx:end_idx+look_ahead]
        trend_resumed = self._check_trend_resumption(future_data, pattern_type)
        
        # Check if stop loss or targets were hit
        target_results = self._check_targets(future_data, pattern_type, end_price, 
                                           stop_loss, target_1r, target_2r)
        
        # Calculate recovery metrics if trend resumed
        recovery_bars = None
        recovery_price = None
        if trend_resumed:
            recovery_info = self._calculate_recovery(future_data, pattern_type, end_price)
            recovery_bars = recovery_info['bars']
            recovery_price = recovery_info['price']
        
        return {
            'start_time': pattern_data.index[0],
            'end_time': pattern_data.index[-1],
            'pattern_type': pattern_type,
            'duration_bars': duration,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'price_range': price_range,
            'price_range_pct': price_range_pct,
            'start_price': start_price,
            'end_price': end_price,
            'risk_amount': risk,
            'risk_pct': risk_pct,
            'potential_reward': potential_reward,
            'reward_pct': reward_pct,
            'risk_reward_ratio': risk_reward_ratio,
            'stop_loss': stop_loss,
            'target_1r': target_1r,
            'target_2r': target_2r,
            'stop_hit': target_results['stop_hit'],
            'target_1r_hit': target_results['target_1r_hit'],
            'target_2r_hit': target_results['target_2r_hit'],
            'bars_to_target': target_results['bars_to_target'],
            'avg_volume': avg_volume,
            'volume_change_pct': volume_change_pct,
            'pattern_strength': strength,
            'outcome': 'resumed' if trend_resumed else 'reversed',
            'recovery_bars': recovery_bars,
            'recovery_price': recovery_price
        }
    
    def _calculate_recovery(self, future_data: pd.DataFrame, 
                           pattern_type: str, 
                           end_price: float) -> Dict:
        """Calculate how long it took for trend to recover"""
        for i in range(len(future_data)):
            if pattern_type == 'Pullback':
                # In uptrend, look for new high
                if future_data['High'].iloc[i] > end_price:
                    return {
                        'bars': i + 1,
                        'price': future_data['High'].iloc[i]
                    }
            else:
                # In downtrend, look for new low
                if future_data['Low'].iloc[i] < end_price:
                    return {
                        'bars': i + 1,
                        'price': future_data['Low'].iloc[i]
                    }
        
        return {
            'bars': None,
            'price': None
        }
    
    def _check_trend_resumption(self, future_data: pd.DataFrame, 
                              pattern_type: str) -> bool:
        """Checks if the original trend resumed after countermove"""
        if pattern_type == 'Pullback':
            # In uptrend, check if price made new high
            return future_data['High'].max() > future_data['High'].iloc[0]
        else:
            # In downtrend, check if price made new low
            return future_data['Low'].min() < future_data['Low'].iloc[0]
    
    def _classify_pattern_strength(self, duration: int, 
                                 price_change_pct: float,
                                 price_range_pct: float,
                                 volume_change_pct: float) -> str:
        """
        Classify pattern strength based on key metrics
        """
        # Score different aspects
        duration_score = 0
        if duration >= 4:
            duration_score = 3
        elif duration >= 3:
            duration_score = 2
        else:
            duration_score = 1
        
        price_score = 0
        if abs(price_change_pct) >= 0.75:
            price_score = 3
        elif abs(price_change_pct) >= 0.5:
            price_score = 2
        else:
            price_score = 1
        
        range_score = 0
        if price_range_pct >= 1.0:
            range_score = 3
        elif price_range_pct >= 0.75:
            range_score = 2
        else:
            range_score = 1
        
        volume_score = 0
        if abs(volume_change_pct) >= 100:  # Volume doubled
            volume_score = 3
        elif abs(volume_change_pct) >= 50:
            volume_score = 2
        else:
            volume_score = 1
        
        # Calculate total score
        total_score = duration_score + price_score + range_score + volume_score
        
        # Classify based on total score
        if total_score >= 10:
            return 'Strong'
        elif total_score >= 7:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _calculate_probabilities(self, df: pd.DataFrame) -> Dict:
        """Calculate probability statistics for different pattern strengths"""
        stats = {}
        
        # Calculate stats for each strength level
        for strength in df['pattern_strength'].unique():
            strength_df = df[df['pattern_strength'] == strength]
            
            resumed = strength_df['outcome'] == 'resumed'
            stats[strength] = {
                'total': len(strength_df),
                'resume_prob': resumed.mean(),
                'reversal_prob': (~resumed).mean(),
                'avg_recovery_bars': strength_df[resumed]['recovery_bars'].mean(),
                'avg_price_change_pct': strength_df['price_change_pct'].mean(),
                'avg_volume_change_pct': strength_df['volume_change_pct'].mean()
            }
        
        # Calculate stats by pattern type
        for pattern in ['Rally', 'Pullback']:
            pattern_df = df[df['pattern_type'] == pattern]
            if len(pattern_df) > 0:
                resumed = pattern_df['outcome'] == 'resumed'
                stats[f'{pattern}_stats'] = {
                    'resume_prob': resumed.mean(),
                    'avg_duration': pattern_df['duration_bars'].mean(),
                    'avg_price_change_pct': pattern_df['price_change_pct'].mean(),
                    'success_by_strength': {
                        strength: group['outcome'].value_counts(normalize=True).get('resumed', 0)
                        for strength, group in pattern_df.groupby('pattern_strength')
                    }
                }
        
        return stats
    
    def _check_targets(self, future_data: pd.DataFrame, pattern_type: str,
                      entry_price: float, stop_loss: float, 
                      target_1r: float, target_2r: float) -> Dict:
        """Check if stop loss or profit targets were hit"""
        stop_hit = False
        target_1r_hit = False
        target_2r_hit = False
        bars_to_target = None
        
        for i in range(len(future_data)):
            if pattern_type == 'Pullback':
                # Check stop loss
                if future_data['Low'].iloc[i] <= stop_loss:
                    stop_hit = True
                    bars_to_target = i + 1
                    break
                # Check targets
                if future_data['High'].iloc[i] >= target_1r:
                    target_1r_hit = True
                    if bars_to_target is None:
                        bars_to_target = i + 1
                if future_data['High'].iloc[i] >= target_2r:
                    target_2r_hit = True
                    if bars_to_target is None:
                        bars_to_target = i + 1
            else:  # Rally
                # Check stop loss
                if future_data['High'].iloc[i] >= stop_loss:
                    stop_hit = True
                    bars_to_target = i + 1
                    break
                # Check targets
                if future_data['Low'].iloc[i] <= target_1r:
                    target_1r_hit = True
                    if bars_to_target is None:
                        bars_to_target = i + 1
                if future_data['Low'].iloc[i] <= target_2r:
                    target_2r_hit = True
                    if bars_to_target is None:
                        bars_to_target = i + 1
        
        return {
            'stop_hit': stop_hit,
            'target_1r_hit': target_1r_hit,
            'target_2r_hit': target_2r_hit,
            'bars_to_target': bars_to_target
        }

if __name__ == "__main__":
    analyzer = CountermoveStats()
    countermove_file = "data/countermove_complete/countermove_NNE_trend_data_20241205.csv"
    results = analyzer.analyze_countermove_patterns(countermove_file)