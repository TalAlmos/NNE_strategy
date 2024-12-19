import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class QuickTrendTrader:
    def __init__(self):
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.trades = []
        self.stop_loss_pct = 0.005  # 0.5% stop loss
        self.reversal_start_time = None
        self.max_price_since_entry = 0
        self.min_price_since_entry = float('inf')
        
    def is_countermove(self, data: pd.DataFrame, lookback: int = 3) -> bool:
        """Identify if current down/up move is a countermove rather than true reversal"""
        if len(data) < lookback + 1:
            return False
            
        # Get recent price action
        recent_data = data.tail(lookback + 1)
        initial_move = recent_data['Close'].iloc[1] - recent_data['Close'].iloc[0]
        subsequent_move = recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[1]
        
        # Volume analysis
        initial_volume = recent_data['Volume'].iloc[1]
        avg_subsequent_volume = recent_data['Volume'].iloc[2:].mean()
        
        # Countermove characteristics (updated):
        # 1. Initial strong move
        # 2. Subsequent move against initial direction but with less strength
        # 3. Lower volume on countermove
        # 4. Duration of countermove (should be short)
        if initial_move > 0:  # Upward initial move
            return (subsequent_move < 0 and  # Downward countermove
                    abs(subsequent_move) < abs(initial_move) * 0.8 and  # Less strength (increased threshold)
                    avg_subsequent_volume < initial_volume * 0.7 and  # Lower volume (stricter)
                    len(recent_data[recent_data['Close'].diff() < 0]) <= 2)  # Short duration
        else:  # Downward initial move
            return (subsequent_move > 0 and  # Upward countermove
                    abs(subsequent_move) < abs(initial_move) * 0.8 and  # Less strength
                    avg_subsequent_volume < initial_volume * 0.7 and  # Lower volume
                    len(recent_data[recent_data['Close'].diff() > 0]) <= 2)  # Short duration
    
    def detect_reversal(self, data: pd.DataFrame) -> Dict:
        """Detect if current price action indicates a reversal"""
        price_changes = data['Close'].pct_change()
        volume_ratio = data['Volume'] / data['Volume'].rolling(5).mean()
        
        # Look for initial surge with stronger conditions
        surge_detected = (
            abs(price_changes.iloc[-1]) > 0.003 and  # 0.3% move
            volume_ratio.iloc[-1] > 1.2 and  # Reduced volume threshold for faster detection
            # Check if move is continuing the surge direction
            (price_changes.iloc[-1] > 0 and data['Close'].iloc[-1] > data['Close'].iloc[-2]) or
            (price_changes.iloc[-1] < 0 and data['Close'].iloc[-1] < data['Close'].iloc[-2])
        )
        
        # Determine direction
        if surge_detected:
            direction = 'UP' if price_changes.iloc[-1] > 0 else 'DOWN'
            return {
                'detected': True,
                'direction': direction,
                'strength': abs(price_changes.iloc[-1]),
                'volume_ratio': volume_ratio.iloc[-1],
                'price': data['Close'].iloc[-1]
            }
        
        return {'detected': False}
    
    def quick_trend_analyzer(self, data: pd.DataFrame) -> Dict:
        """Fast-acting trend analyzer with countermove pattern recognition"""
        # Basic indicators
        price_changes = data['Close'].pct_change()
        volume_ratio = data['Volume'] / data['Volume'].rolling(5).mean()
        momentum = price_changes.rolling(3).sum()
        
        # Detect potential reversal
        reversal = self.detect_reversal(data)
        
        # If we see a potential reversal, check if subsequent moves are countermoves
        if reversal['detected'] and len(data) >= 4:
            is_counter = self.is_countermove(data.tail(4))
            if is_counter:
                # Ignore the countermove, maintain previous trend
                reversal['detected'] = False
        
        # Entry conditions
        entry_conditions = {
            'LONG': (
                reversal['detected'] and
                reversal['direction'] == 'UP' and
                not self.is_countermove(data)
            ),
            'SHORT': (
                reversal['detected'] and
                reversal['direction'] == 'DOWN' and
                not self.is_countermove(data)
            )
        }
        
        return {
            'entry_conditions': entry_conditions,
            'reversal': reversal,
            'current_price': data['Close'].iloc[-1],
            'current_time': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Datetime'].iloc[-1])
        }
    
    def should_exit_position(self, current_price: float, current_time: datetime, signals: Dict) -> Tuple[bool, str]:
        """Determine if we should exit the position"""
        if self.position is None:
            return False, ""
            
        # Update max/min prices
        if self.position == 'LONG':
            self.max_price_since_entry = max(self.max_price_since_entry, current_price)
            # Exit immediately if we detect a strong reversal
            if (signals['reversal']['detected'] and 
                signals['reversal']['direction'] == 'DOWN' and
                signals['reversal']['strength'] > 0.005):  # Strong reversal
                return True, "Strong Reversal"
            # Exit on trailing stop
            if current_price < self.max_price_since_entry * 0.995:
                return True, "Trailing Stop"
                
        else:  # SHORT position
            self.min_price_since_entry = min(self.min_price_since_entry, current_price)
            # Exit immediately if we detect a strong reversal
            if (signals['reversal']['detected'] and 
                signals['reversal']['direction'] == 'UP' and
                signals['reversal']['strength'] > 0.005):  # Strong reversal
                return True, "Strong Reversal"
            # Exit on trailing stop
            if current_price > self.min_price_since_entry * 1.005:
                return True, "Trailing Stop"
        
        return False, ""
    
    def execute_trades(self, data: pd.DataFrame) -> List[Dict]:
        """Execute trades with improved entry/exit logic"""
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.trades = []
        self.max_price_since_entry = 0
        self.min_price_since_entry = float('inf')
        
        for i in range(5, len(data)):
            current_data = data.iloc[:i+1].copy()
            signals = self.quick_trend_analyzer(current_data)
            
            current_time = signals['current_time']
            current_price = signals['current_price']
            
            # Check exit conditions if in position
            if self.position:
                should_exit, reason = self.should_exit_position(current_price, current_time, signals)
                if should_exit:
                    self.close_position(current_price, current_time, reason)
                    continue
            
            # Check entry conditions if not in position
            if self.position is None:
                # If we detect a reversal
                if signals['reversal']['detected']:
                    # Update reversal start time if not set or if it's a new reversal
                    if (self.reversal_start_time is None or 
                        (current_time - self.reversal_start_time).total_seconds() >= 180):
                        self.reversal_start_time = current_time
                
                # Check if we should enter based on reversal confirmation
                if self.reversal_start_time:
                    time_since_reversal = (current_time - self.reversal_start_time).total_seconds()
                    if time_since_reversal >= 180:  # 3 minutes rule
                        if signals['entry_conditions']['LONG']:
                            self.enter_position('LONG', current_price, current_time)
                        elif signals['entry_conditions']['SHORT']:
                            self.enter_position('SHORT', current_price, current_time)
    
    def enter_position(self, position_type: str, price: float, time: datetime):
        """Enter a new position"""
        self.position = position_type
        self.entry_price = price
        self.entry_time = time
    
    def close_position(self, price: float, time: datetime, reason: str):
        """Close current position"""
        if self.position == 'LONG':
            profit = price - self.entry_price
        else:  # SHORT
            profit = self.entry_price - price
        
        self.trades.append({
            'position': self.position,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'exit_time': time,
            'exit_price': price,
            'profit': profit,
            'profit_pct': (profit / self.entry_price) * 100,
            'duration': (time - self.entry_time).total_seconds() / 60,
            'reason': reason
        })
        
        self.position = None
        self.entry_price = 0
        self.entry_time = None
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss has been hit"""
        if self.position == 'LONG':
            return current_price < self.entry_price * (1 - self.stop_loss_pct)
        else:  # SHORT
            return current_price > self.entry_price * (1 + self.stop_loss_pct)
    
    def analyze_trade_conditions(self, data: pd.DataFrame, target_time: str, 
                               pre_trade_minutes: int = 5, during_trade_minutes: int = 12):
        """Analyze conditions before and during a specific trade"""
        target_idx = data[data['Datetime'].dt.strftime('%H:%M:%S') == target_time].index[0]
        
        print(f"\nAnalyzing conditions around {target_time} trade:")
        
        # Look at minutes before the trade
        print("\nPre-Entry Conditions:")
        for i in range(target_idx - pre_trade_minutes, target_idx + 1):
            current_data = data.iloc[:i+1].copy()
            current_time = current_data.iloc[-1]['Datetime']
            
            # Calculate indicators
            price_changes = current_data['Close'].pct_change()
            volume_ratio = current_data['Volume'] / current_data['Volume'].rolling(5).mean()
            higher_lows = (current_data['Low'] > current_data['Low'].shift(1)).astype(int)
            momentum = price_changes.rolling(3).sum()
            
            print(f"\nTime: {current_time.strftime('%H:%M:%S')}")
            print(f"Price: ${current_data.iloc[-1]['Close']:.2f}")
            print(f"Volume: {current_data.iloc[-1]['Volume']:,}")
            print(f"Volume Ratio: {volume_ratio.iloc[-1]:.2f}")
            print(f"Higher Lows (last 3): {higher_lows.rolling(3).sum().iloc[-1]:.0f}")
            print(f"Momentum: {momentum.iloc[-1]:.3%}")
            
            # Check entry conditions
            long_condition = (
                higher_lows.rolling(3).sum().iloc[-1] >= 2 and
                volume_ratio.iloc[-1] > 1.2 and
                momentum.iloc[-1] > 0
            )
            print(f"Long Signal: {long_condition}")
            
            # Check countermove
            if len(current_data) >= 4:
                is_counter = self.is_countermove(current_data.tail(4))
                print(f"Is Countermove: {is_counter}")
        
        print("\nDuring Trade Conditions:")
        for i in range(target_idx + 1, target_idx + during_trade_minutes + 1):
            if i >= len(data):
                break
                
            current_data = data.iloc[:i+1].copy()
            current_time = current_data.iloc[-1]['Datetime']
            
            # Calculate indicators
            price_changes = current_data['Close'].pct_change()
            current_price = current_data.iloc[-1]['Close']
            entry_price = data.iloc[target_idx]['Close']
            price_change_pct = (current_price - entry_price) / entry_price * 100
            volume_ratio = current_data['Volume'] / current_data['Volume'].rolling(5).mean()
            momentum = price_changes.rolling(3).sum()
            
            print(f"\nTime: {current_time.strftime('%H:%M:%S')}")
            print(f"Price: ${current_price:.2f}")
            print(f"Change from Entry: {price_change_pct:.2f}%")
            print(f"Volume Ratio: {volume_ratio.iloc[-1]:.2f}")
            print(f"Momentum: {momentum.iloc[-1]:.3%}")
            
            # Check reversal conditions
            reversal = self.detect_reversal(current_data)
            print(f"Reversal Detected: {reversal['detected']}")
            if reversal['detected']:
                print(f"Reversal Direction: {reversal['direction']}")
                print(f"Reversal Strength: {reversal['strength']:.3%}")
            
            # Check if it's a countermove
            if len(current_data) >= 4:
                is_counter = self.is_countermove(current_data.tail(4))
                print(f"Is Countermove: {is_counter}")

def main():
    # Initialize trader
    trader = QuickTrendTrader()
    
    # Process December 5th data
    file = Path("nne_strategy/data/raw/NNE_data_20241205.csv")
    
    # Load and prepare data
    df = pd.read_csv(file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Analyze pre-entry conditions for 10:08 trade
    print("\nAnalyzing conditions before 10:08 trade:")
    target_time = "10:08:00"
    pre_trade_data = df[df['Datetime'].dt.strftime('%H:%M:%S') <= target_time].copy()
    
    # Look at the last 10 minutes before entry
    start_idx = len(pre_trade_data) - 10
    for i in range(start_idx, len(pre_trade_data)):
        current_time = pre_trade_data.iloc[i]['Datetime']
        current_price = pre_trade_data.iloc[i]['Close']
        current_volume = pre_trade_data.iloc[i]['Volume']
        
        # Calculate price change
        if i > 0:
            price_change = current_price - pre_trade_data.iloc[i-1]['Close']
            price_change_pct = (price_change / pre_trade_data.iloc[i-1]['Close']) * 100
            volume_ratio = current_volume / pre_trade_data.iloc[max(0, i-5):i]['Volume'].mean()
            print(f"\nTime: {current_time.strftime('%H:%M:%S')}")
            print(f"Price: ${current_price:.2f} ({price_change_pct:+.2f}%)")
            print(f"Volume: {current_volume:,}")
            print(f"Volume Ratio: {volume_ratio:.2f}")
            
            # Check entry conditions
            data_slice = pre_trade_data.iloc[:i+1].copy()
            signals = trader.quick_trend_analyzer(data_slice)
            print(f"Entry Signal LONG: {signals['entry_conditions']['LONG']}")
            if signals['reversal']['detected']:
                print(f"Reversal Direction: {signals['reversal']['direction']}")
    
    # Execute trades for this day
    day_trades = trader.execute_trades(df)
    
    # Print trades in tabular format
    if day_trades:
        print("\nTrade Summary:")
        print("|Long or Short|Entry Time|Entry Price|Exit Time|Exit Price|Profit|%|")
        print("|-------------|-----------|------------|----------|-----------|------|---|")
        for trade in day_trades:
            print(f"|{trade['position']}|{trade['entry_time'].strftime('%H:%M:%S')}|${trade['entry_price']:.2f}|{trade['exit_time'].strftime('%H:%M:%S')}|${trade['exit_price']:.2f}|${trade['profit']:.2f}|{trade['profit_pct']:.2f}%|")
        
        # Print daily summary
        daily_profit = sum(trade['profit'] for trade in day_trades)
        win_rate = sum(1 for trade in day_trades if trade['profit'] > 0) / len(day_trades) * 100
        avg_duration = sum(trade['duration'] for trade in day_trades) / len(day_trades)
        print(f"\nDaily Summary:")
        print(f"Total Trades: {len(day_trades)}")
        print(f"Total Profit/Loss: ${daily_profit:.2f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Duration: {avg_duration:.1f} minutes")

if __name__ == "__main__":
    main() 