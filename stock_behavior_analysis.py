import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class TimePattern:
    period: str
    success_rate: float
    avg_volume: float
    avg_price_change: float
    trend_direction: str
    confidence: float

@dataclass
class PricePattern:
    pattern_type: str
    avg_magnitude: float
    success_rate: float
    typical_duration: float
    volume_characteristic: str
    key_levels: List[float]

@dataclass
class VolumePattern:
    period: str
    avg_volume: float
    price_correlation: float
    trend_indication: str
    significance: float

@dataclass
class TrendPattern:
    direction: str
    avg_duration: float
    avg_magnitude: float
    success_rate: float
    typical_volume: float
    common_reversal_points: List[float]

class StockBehaviorAnalysis:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.time_patterns = {}
        self.price_patterns = {}
        self.volume_patterns = {}
        self.trend_patterns = {}
        self.daily_statistics = {}
        
    def analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, TimePattern]:
        """Analyze patterns based on time of day."""
        patterns = {}
        
        # Opening hour pattern (9:30-10:30)
        opening_data = df[df['Datetime'].dt.hour == 9]
        opening_pattern = TimePattern(
            period="Opening Hour",
            success_rate=self._calculate_success_rate(opening_data),
            avg_volume=opening_data['Volume'].mean(),
            avg_price_change=self._calculate_price_change(opening_data),
            trend_direction=self._determine_trend_direction(opening_data),
            confidence=self._calculate_confidence(opening_data)
        )
        patterns["opening"] = opening_pattern
        
        # Mid-day pattern (11:00-14:00)
        midday_data = df[df['Datetime'].dt.hour.isin([11, 12, 13])]
        midday_pattern = TimePattern(
            period="Mid-Day",
            success_rate=self._calculate_success_rate(midday_data),
            avg_volume=midday_data['Volume'].mean(),
            avg_price_change=self._calculate_price_change(midday_data),
            trend_direction=self._determine_trend_direction(midday_data),
            confidence=self._calculate_confidence(midday_data)
        )
        patterns["midday"] = midday_pattern
        
        # Closing hour pattern (15:00-16:00)
        closing_data = df[df['Datetime'].dt.hour == 15]
        closing_pattern = TimePattern(
            period="Closing Hour",
            success_rate=self._calculate_success_rate(closing_data),
            avg_volume=closing_data['Volume'].mean(),
            avg_price_change=self._calculate_price_change(closing_data),
            trend_direction=self._determine_trend_direction(closing_data),
            confidence=self._calculate_confidence(closing_data)
        )
        patterns["closing"] = closing_pattern
        
        return patterns
    
    def analyze_price_patterns(self, df: pd.DataFrame) -> Dict[str, PricePattern]:
        """Analyze price action patterns."""
        patterns = {}
        
        # Trend following patterns
        trend_pattern = PricePattern(
            pattern_type="Trend Following",
            avg_magnitude=self._calculate_trend_magnitude(df),
            success_rate=self._calculate_trend_success_rate(df),
            typical_duration=self._calculate_trend_duration(df),
            volume_characteristic=self._analyze_volume_characteristic(df),
            key_levels=self._identify_key_levels(df)
        )
        patterns["trend"] = trend_pattern
        
        # Reversal patterns
        reversal_pattern = PricePattern(
            pattern_type="Reversal",
            avg_magnitude=self._calculate_reversal_magnitude(df),
            success_rate=self._calculate_reversal_success_rate(df),
            typical_duration=self._calculate_reversal_duration(df),
            volume_characteristic=self._analyze_reversal_volume(df),
            key_levels=self._identify_reversal_levels(df)
        )
        patterns["reversal"] = reversal_pattern
        
        return patterns
    
    def analyze_volume_patterns(self, df: pd.DataFrame) -> Dict[str, VolumePattern]:
        """Analyze volume patterns and their relationship with price."""
        patterns = {}
        
        # High volume pattern
        high_volume_data = df[df['Volume'] > df['Volume'].mean() * 1.5]
        high_volume_pattern = VolumePattern(
            period="High Volume",
            avg_volume=high_volume_data['Volume'].mean(),
            price_correlation=self._calculate_volume_price_correlation(high_volume_data),
            trend_indication=self._analyze_volume_trend_indication(high_volume_data),
            significance=self._calculate_volume_significance(high_volume_data)
        )
        patterns["high_volume"] = high_volume_pattern
        
        # Low volume pattern
        low_volume_data = df[df['Volume'] < df['Volume'].mean() * 0.5]
        low_volume_pattern = VolumePattern(
            period="Low Volume",
            avg_volume=low_volume_data['Volume'].mean(),
            price_correlation=self._calculate_volume_price_correlation(low_volume_data),
            trend_indication=self._analyze_volume_trend_indication(low_volume_data),
            significance=self._calculate_volume_significance(low_volume_data)
        )
        patterns["low_volume"] = low_volume_pattern
        
        return patterns
    
    def analyze_trend_patterns(self, df: pd.DataFrame) -> Dict[str, TrendPattern]:
        """Analyze trend patterns and characteristics."""
        patterns = {}
        
        # Uptrend pattern
        uptrend_data = self._identify_uptrends(df)
        uptrend_pattern = TrendPattern(
            direction="Up",
            avg_duration=self._calculate_trend_duration(uptrend_data),
            avg_magnitude=self._calculate_trend_magnitude(uptrend_data),
            success_rate=self._calculate_trend_success_rate(uptrend_data),
            typical_volume=self._calculate_typical_volume(uptrend_data),
            common_reversal_points=self._identify_reversal_points(uptrend_data)
        )
        patterns["uptrend"] = uptrend_pattern
        
        # Downtrend pattern
        downtrend_data = self._identify_downtrends(df)
        downtrend_pattern = TrendPattern(
            direction="Down",
            avg_duration=self._calculate_trend_duration(downtrend_data),
            avg_magnitude=self._calculate_trend_magnitude(downtrend_data),
            success_rate=self._calculate_trend_success_rate(downtrend_data),
            typical_volume=self._calculate_typical_volume(downtrend_data),
            common_reversal_points=self._identify_reversal_points(downtrend_data)
        )
        patterns["downtrend"] = downtrend_pattern
        
        return patterns
    
    def analyze_daily_behavior(self, df: pd.DataFrame) -> None:
        """Analyze and store daily behavioral patterns."""
        self.time_patterns = self.analyze_time_patterns(df)
        self.price_patterns = self.analyze_price_patterns(df)
        self.volume_patterns = self.analyze_volume_patterns(df)
        self.trend_patterns = self.analyze_trend_patterns(df)
        
        # Calculate daily statistics
        self.daily_statistics = {
            'avg_daily_range': self._calculate_daily_range(df),
            'avg_trend_duration': self._calculate_avg_trend_duration(df),
            'common_support_levels': self._identify_support_levels(df),
            'common_resistance_levels': self._identify_resistance_levels(df),
            'volume_profile': self._analyze_volume_profile(df)
        }
    
    def get_real_time_signals(self, current_data: pd.DataFrame) -> Dict:
        """Generate trading signals based on current market conditions."""
        if len(current_data) < 5:
            return {'action': "HOLD", 'confidence': 0.0}

        # Calculate recent price movement
        recent_prices = current_data['Close'].tail(5)
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        # Calculate volume trend
        recent_volume = current_data['Volume'].tail(5)
        volume_trend = recent_volume.mean() / current_data['Volume'].mean() if current_data['Volume'].mean() > 0 else 0
        
        # Calculate momentum
        momentum = current_data['Close'].pct_change().tail(5).mean()
        
        # Calculate trend strength
        trend_strength = abs(momentum) * volume_trend * (1 + abs(price_change))
        if trend_strength > 0:
            trend_strength = max(1.0, trend_strength * 10)  # Scale up non-zero trend strength
        
        # Calculate confidence
        confidence = self._calculate_confidence(current_data)
        
        # Determine action based on price movement and volume
        if price_change > 0.002 and volume_trend > 1.1 and momentum > 0:  # 0.2% move up with higher volume
            action = "STRONG_BUY" if confidence > 0.7 else "BUY"
        elif price_change < -0.002 and volume_trend > 1.1 and momentum < 0:  # 0.2% move down with higher volume
            action = "STRONG_SELL" if confidence > 0.7 else "SELL"
        else:
            action = "HOLD"
        
        return {
            'action': action,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'supporting_patterns': {
                'price_change': price_change * 100,  # Convert to percentage
                'volume_trend': volume_trend,
                'momentum': momentum * 100  # Convert to percentage
            }
        }
    
    # Helper methods for pattern analysis
    def _calculate_success_rate(self, df: pd.DataFrame) -> float:
        """Calculate success rate of patterns in the given data."""
        if len(df) < 2:
            return 0.0
        price_changes = df['Close'].pct_change()
        return (price_changes > 0).mean() * 100
    
    def _calculate_price_change(self, df: pd.DataFrame) -> float:
        """Calculate average price change."""
        if len(df) < 2:
            return 0.0
        return df['Close'].pct_change().mean() * 100
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine predominant trend direction."""
        if len(df) < 2:
            return "Insufficient data"
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
        return "Up" if price_change > 0 else "Down"
    
    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence level of the pattern."""
        if len(df) < 5:  # Need at least 5 minutes of data
            return 0.0
        
        # Calculate short-term momentum
        price_changes = df['Close'].pct_change()
        recent_momentum = price_changes.tail(5).mean()
        
        # Calculate volume trend
        volume_trend = df['Volume'].tail(5).mean() / df['Volume'].mean() if df['Volume'].mean() > 0 else 0
        
        # Calculate price volatility
        volatility = price_changes.std()
        
        # Combine factors
        momentum_score = abs(recent_momentum) / (volatility if volatility > 0 else 0.0001)
        volume_score = min(volume_trend, 2.0) / 2.0  # Normalize volume score
        
        confidence = (momentum_score * 0.7 + volume_score * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_trend_magnitude(self, df: pd.DataFrame) -> float:
        """Calculate average trend magnitude."""
        return abs(df['Close'].pct_change()).mean() * 100
    
    def _identify_key_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify key price levels."""
        # Implementation for identifying support/resistance
        return []
    
    def _calculate_volume_price_correlation(self, df: pd.DataFrame) -> float:
        """Calculate correlation between volume and price changes."""
        price_changes = df['Close'].pct_change()
        return df['Volume'].corr(price_changes)
    
    def _identify_uptrends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify periods of uptrends."""
        # Implementation for uptrend identification
        return df
    
    def _identify_downtrends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify periods of downtrends."""
        # Implementation for downtrend identification
        return df
    
    def _calculate_daily_range(self, df: pd.DataFrame) -> float:
        """Calculate average daily price range."""
        return (df['High'] - df['Low']).mean()
    
    def _identify_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify common support levels."""
        # Implementation for support level identification
        return []
    
    def _identify_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify common resistance levels."""
        # Implementation for resistance level identification
        return []
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume distribution throughout the day."""
        return {
            'morning_volume': df[df['Datetime'].dt.hour < 11]['Volume'].mean(),
            'midday_volume': df[df['Datetime'].dt.hour.isin([11, 12, 13])]['Volume'].mean(),
            'closing_volume': df[df['Datetime'].dt.hour > 13]['Volume'].mean()
        }
    
    def _match_time_pattern(self, current_time: datetime) -> Dict:
        """Match current time with known time patterns."""
        hour = current_time.hour
        if hour < 11:
            return self.time_patterns.get("opening")
        elif hour > 13:
            return self.time_patterns.get("closing")
        return self.time_patterns.get("midday")
    
    def _match_price_pattern(self, current_data: pd.DataFrame) -> Dict:
        """Match current price action with known patterns."""
        # Implementation for pattern matching
        return {}
    
    def _match_volume_pattern(self, current_volume: float) -> Dict:
        """Match current volume with known patterns."""
        # Implementation for volume pattern matching
        return {}
    
    def _match_trend_pattern(self, current_data: pd.DataFrame) -> Dict:
        """Match current trend with known patterns."""
        # Implementation for trend pattern matching
        return {}
    
    def _calculate_signal_confidence(self, time_match: Dict, price_match: Dict,
                                   volume_match: Dict, trend_match: Dict) -> float:
        """Calculate overall confidence in the signal."""
        # Implementation for confidence calculation
        return 0.0
    
    def _determine_action(self, confidence: float) -> str:
        """Determine trading action based on confidence level."""
        if confidence > 0.8:
            return "STRONG_BUY"
        elif confidence > 0.6:
            return "BUY"
        elif confidence < 0.2:
            return "STRONG_SELL"
        elif confidence < 0.4:
            return "SELL"
        return "HOLD"
    
    def _calculate_risk_levels(self, current_price: float) -> Dict:
        """Calculate appropriate risk levels for the trade."""
        return {
            'stop_loss': current_price * 0.99,  # Example: 1% stop loss
            'target': current_price * 1.02      # Example: 2% target
        }
    
    def _calculate_trend_success_rate(self, df: pd.DataFrame) -> float:
        """Calculate success rate of trend following."""
        if len(df) < 2:
            return 0.0
        
        # Calculate consecutive moves in same direction
        price_changes = df['Close'].pct_change()
        consecutive_moves = (price_changes > 0).astype(int).groupby(
            (price_changes > 0).ne((price_changes > 0).shift()).cumsum()
        ).count()
        
        # Success rate is percentage of moves that continued
        return (consecutive_moves > 1).mean() * 100

    def _calculate_trend_duration(self, df: pd.DataFrame) -> float:
        """Calculate average duration of trends."""
        if len(df) < 2:
            return 0.0
        
        # Group consecutive price moves
        price_changes = df['Close'].pct_change()
        trend_groups = (price_changes > 0).ne((price_changes > 0).shift()).cumsum()
        
        # Calculate duration of each trend
        durations = trend_groups.value_counts()
        return durations.mean()

    def _analyze_volume_characteristic(self, df: pd.DataFrame) -> str:
        """Analyze volume characteristics during price moves."""
        if len(df) < 2:
            return "Insufficient data"
        
        # Compare volume on up vs down moves
        price_changes = df['Close'].pct_change()
        up_volume = df[price_changes > 0]['Volume'].mean()
        down_volume = df[price_changes < 0]['Volume'].mean()
        
        if up_volume > down_volume * 1.2:
            return "Higher volume on up moves"
        elif down_volume > up_volume * 1.2:
            return "Higher volume on down moves"
        return "Balanced volume"

    def _calculate_reversal_magnitude(self, df: pd.DataFrame) -> float:
        """Calculate average magnitude of price reversals."""
        if len(df) < 2:
            return 0.0
        
        # Find points where price direction changes
        price_changes = df['Close'].pct_change()
        reversals = price_changes[price_changes * price_changes.shift() < 0]
        
        return abs(reversals).mean() * 100

    def _calculate_reversal_success_rate(self, df: pd.DataFrame) -> float:
        """Calculate success rate of reversal patterns."""
        if len(df) < 3:
            return 0.0
        
        # Find reversals and check if they led to sustained moves
        price_changes = df['Close'].pct_change()
        reversals = (price_changes * price_changes.shift() < 0)
        
        success_count = 0
        total_count = 0
        
        for i in range(1, len(df)-1):
            if reversals.iloc[i]:
                total_count += 1
                # Check if reversal led to continued move
                if (price_changes.iloc[i] * price_changes.iloc[i+1] > 0):
                    success_count += 1
        
        return (success_count / total_count * 100) if total_count > 0 else 0.0

    def _calculate_reversal_duration(self, df: pd.DataFrame) -> float:
        """Calculate average duration of reversal moves."""
        if len(df) < 3:
            return 0.0
        
        price_changes = df['Close'].pct_change()
        reversals = (price_changes * price_changes.shift() < 0)
        reversal_groups = reversals.cumsum()
        
        return reversal_groups.value_counts().mean()

    def _analyze_reversal_volume(self, df: pd.DataFrame) -> str:
        """Analyze volume characteristics during reversals."""
        if len(df) < 2:
            return "Insufficient data"
        
        price_changes = df['Close'].pct_change()
        reversals = (price_changes * price_changes.shift() < 0)
        
        reversal_volume = df[reversals]['Volume'].mean()
        normal_volume = df[~reversals]['Volume'].mean()
        
        if reversal_volume > normal_volume * 1.5:
            return "High volume reversals"
        elif reversal_volume < normal_volume * 0.8:
            return "Low volume reversals"
        return "Normal volume reversals"

    def _identify_reversal_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify price levels where reversals commonly occur."""
        if len(df) < 2:
            return []
        
        price_changes = df['Close'].pct_change()
        reversals = df[price_changes * price_changes.shift() < 0]['Close']
        
        # Use kernel density estimation to find common reversal levels
        if len(reversals) > 0:
            kde = np.histogram(reversals, bins=10)
            return list(kde[1][kde[0].argsort()[-3:]])  # Top 3 reversal levels
        return []

    def _calculate_typical_volume(self, df: pd.DataFrame) -> float:
        """Calculate typical volume during trends."""
        return df['Volume'].mean()

    def _identify_reversal_points(self, df: pd.DataFrame) -> List[float]:
        """Identify common price levels where trends reverse."""
        if len(df) < 2:
            return []
        
        # Find local maxima and minima
        price_changes = df['Close'].pct_change()
        reversals = df[price_changes * price_changes.shift() < 0]['Close']
        
        if len(reversals) > 0:
            # Return most common reversal points
            return list(reversals.value_counts().head(3).index)
        return []

    def _calculate_avg_trend_duration(self, df: pd.DataFrame) -> float:
        """Calculate average duration of trends in minutes."""
        if len(df) < 2:
            return 0.0
        
        price_changes = df['Close'].pct_change()
        trend_groups = (price_changes > 0).ne((price_changes > 0).shift()).cumsum()
        
        return trend_groups.value_counts().mean()

    def _analyze_volume_trend_indication(self, df: pd.DataFrame) -> str:
        """Analyze how volume indicates trend strength."""
        if len(df) < 2:
            return "Insufficient data"
        
        price_changes = df['Close'].pct_change()
        volume_price_corr = df['Volume'].corr(abs(price_changes))
        
        if volume_price_corr > 0.5:
            return "Strong trend confirmation"
        elif volume_price_corr > 0.2:
            return "Moderate trend confirmation"
        return "Weak trend confirmation"

    def _calculate_volume_significance(self, df: pd.DataFrame) -> float:
        """Calculate significance of volume patterns."""
        if len(df) < 2:
            return 0.0
        
        # Compare volume to price movement
        price_changes = df['Close'].pct_change()
        volume_price_impact = (abs(price_changes) * df['Volume']).mean()
        
        return volume_price_impact * 100

def main():
    # Example usage
    analyzer = StockBehaviorAnalysis("NNE")
    
    # Read historical data
    df = pd.read_csv('nne_strategy/data/analysis/trends/NNE_trend_data_20241205.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Analyze patterns
    analyzer.analyze_daily_behavior(df)
    
    # Get current signals
    current_data = df.tail(10)  # Last 10 minutes of data
    signals = analyzer.get_real_time_signals(current_data)
    
    # Print analysis results
    print("\nStock Behavior Analysis Results")
    print("=" * 50)
    print(f"\nTime Patterns:")
    for period, pattern in analyzer.time_patterns.items():
        print(f"\n{period.upper()}:")
        print(f"Success Rate: {pattern.success_rate:.2f}%")
        print(f"Avg Volume: {pattern.avg_volume:,.0f}")
        print(f"Avg Price Change: {pattern.avg_price_change:.2f}%")
        print(f"Trend Direction: {pattern.trend_direction}")
    
    print(f"\nCurrent Trading Signals:")
    print(f"Action: {signals['action']}")
    print(f"Confidence: {signals['confidence']:.2f}")
    print(f"Stop Loss: ${signals['risk_levels']['stop_loss']:.2f}")
    print(f"Target: ${signals['risk_levels']['target']:.2f}")

if __name__ == "__main__":
    main() 