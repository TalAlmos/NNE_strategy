import pandas as pd
import numpy as np
from datetime import datetime

def analyze_major_trends(csv_file, min_price_change_pct=1.0, min_duration_minutes=15, volume_threshold_percentile=75):
    """
    Analyze major market trends from price data.
    
    Parameters:
    - csv_file: Path to CSV file containing price data
    - min_price_change_pct: Minimum price change percentage to consider as significant
    - min_duration_minutes: Minimum duration in minutes for a trend
    - volume_threshold_percentile: Percentile threshold for high volume
    """
    # Read the data
    df = pd.read_csv(csv_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Calculate additional metrics
    df['Price_Change_Pct'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df['High_Volume'] = df['Volume'] > df['Volume'].quantile(volume_threshold_percentile/100)
    
    # Initialize trend analysis
    trends = []
    current_trend = {
        'start_time': df['Datetime'].iloc[0],
        'start_price': df['Close'].iloc[0],
        'direction': None,
        'high': df['Close'].iloc[0],
        'low': df['Close'].iloc[0]
    }
    
    # Analyze price movements
    for i in range(1, len(df)):
        price_change_from_start = ((df['Close'].iloc[i] - current_trend['start_price']) / 
                                 current_trend['start_price'] * 100)
        duration = (df['Datetime'].iloc[i] - current_trend['start_time']).total_seconds() / 60
        
        # Update high and low
        current_trend['high'] = max(current_trend['high'], df['Close'].iloc[i])
        current_trend['low'] = min(current_trend['low'], df['Close'].iloc[i])
        
        # Check if we need to mark a new trend
        if abs(price_change_from_start) >= min_price_change_pct and duration >= min_duration_minutes:
            if current_trend['direction'] is None:
                current_trend['direction'] = 'Up' if price_change_from_start > 0 else 'Down'
            
            # Check if trend has reversed
            if ((current_trend['direction'] == 'Up' and 
                 df['Close'].iloc[i] < current_trend['high'] * (1 - min_price_change_pct/100)) or
                (current_trend['direction'] == 'Down' and 
                 df['Close'].iloc[i] > current_trend['low'] * (1 + min_price_change_pct/100))):
                
                # Record the completed trend
                trends.append({
                    'start_time': current_trend['start_time'],
                    'end_time': df['Datetime'].iloc[i],
                    'start_price': current_trend['start_price'],
                    'end_price': df['Close'].iloc[i],
                    'direction': current_trend['direction'],
                    'duration_minutes': duration,
                    'price_change_pct': price_change_from_start,
                    'high': current_trend['high'],
                    'low': current_trend['low']
                })
                
                # Start new trend
                current_trend = {
                    'start_time': df['Datetime'].iloc[i],
                    'start_price': df['Close'].iloc[i],
                    'direction': None,
                    'high': df['Close'].iloc[i],
                    'low': df['Close'].iloc[i]
                }
    
    # Add the last trend if it meets criteria
    if current_trend['direction'] is not None:
        final_duration = (df['Datetime'].iloc[-1] - current_trend['start_time']).total_seconds() / 60
        final_change = ((df['Close'].iloc[-1] - current_trend['start_price']) / 
                       current_trend['start_price'] * 100)
        
        if abs(final_change) >= min_price_change_pct and final_duration >= min_duration_minutes:
            trends.append({
                'start_time': current_trend['start_time'],
                'end_time': df['Datetime'].iloc[-1],
                'start_price': current_trend['start_price'],
                'end_price': df['Close'].iloc[-1],
                'direction': current_trend['direction'],
                'duration_minutes': final_duration,
                'price_change_pct': final_change,
                'high': current_trend['high'],
                'low': current_trend['low']
            })
    
    return pd.DataFrame(trends)

def analyze_countermoves(df, current_trend, countermove_threshold_pct=0.5, max_countermove_duration=15):
    """
    Analyze countermoves within a trend to distinguish from reversals.
    
    Parameters:
    - countermove_threshold_pct: Maximum allowed countermove percentage
    - max_countermove_duration: Maximum duration (minutes) for a countermove
    """
    countermoves = []
    in_countermove = False
    countermove_start_idx = 0
    
    for i in range(1, len(df)):
        price_change_from_prev = ((df['Close'].iloc[i] - df['Close'].iloc[i-1]) / 
                                df['Close'].iloc[i-1] * 100)
        
        # Check if price is moving against the trend
        is_counter = ((current_trend == 'Up' and price_change_from_prev < 0) or
                     (current_trend == 'Down' and price_change_from_prev > 0))
        
        if is_counter and not in_countermove:
            # Start of countermove
            in_countermove = True
            countermove_start_idx = i
            countermove_start_price = df['Close'].iloc[i-1]
            countermove_start_time = df['Datetime'].iloc[i-1]
        
        elif in_countermove:
            duration = (df['Datetime'].iloc[i] - countermove_start_time).total_seconds() / 60
            total_move = ((df['Close'].iloc[i] - countermove_start_price) / 
                         countermove_start_price * 100)
            
            # Check if countermove has ended
            if ((current_trend == 'Up' and price_change_from_prev > 0) or
                (current_trend == 'Down' and price_change_from_prev < 0)):
                
                countermoves.append({
                    'start_time': countermove_start_time,
                    'end_time': df['Datetime'].iloc[i],
                    'duration': duration,
                    'price_change_pct': total_move,
                    'volume_ratio': df['Volume'].iloc[i] / df['Volume'].mean(),
                    'is_reversal': abs(total_move) > countermove_threshold_pct and 
                                 duration > max_countermove_duration
                })
                in_countermove = False
            
            # Check if countermove has become a reversal
            elif abs(total_move) > countermove_threshold_pct and duration > max_countermove_duration:
                return True, {
                    'start_time': countermove_start_time,
                    'end_time': df['Datetime'].iloc[i],
                    'duration': duration,
                    'price_change_pct': total_move,
                    'volume_ratio': df['Volume'].iloc[i] / df['Volume'].mean()
                }
    
    return False, countermoves

def get_exit_signal(df, current_index, trend_direction, 
                   stop_loss_pct=1.0, trailing_stop_pct=1.5):
    """
    Determine if position should be closed based on price action.
    
    Returns:
    - should_exit: Boolean
    - reason: String explaining exit reason
    """
    if current_index < 5:  # Need some history
        return False, "Insufficient history"
    
    current_price = df['Close'].iloc[current_index]
    max_price = df['High'].iloc[current_index-5:current_index+1].max()
    min_price = df['Low'].iloc[current_index-5:current_index+1].min()
    
    if trend_direction == 'Up':
        # Check if price broke below trailing stop
        trailing_stop = max_price * (1 - trailing_stop_pct/100)
        if current_price < trailing_stop:
            return True, "Trailing stop triggered"
        
        # Check for reversal pattern
        if (df['Volume'].iloc[current_index] > df['Volume'].mean() * 1.5 and
            current_price < df['Close'].iloc[current_index-1] * (1 - stop_loss_pct/100)):
            return True, "High volume reversal"
            
    else:  # Downtrend
        # Check if price broke above trailing stop
        trailing_stop = min_price * (1 + trailing_stop_pct/100)
        if current_price > trailing_stop:
            return True, "Trailing stop triggered"
        
        # Check for reversal pattern
        if (df['Volume'].iloc[current_index] > df['Volume'].mean() * 1.5 and
            current_price > df['Close'].iloc[current_index-1] * (1 + stop_loss_pct/100)):
            return True, "High volume reversal"
    
    return False, "Continue trend"

def format_trade_signals(df):
    """Format real-time trading signals."""
    signals = []
    current_trend = None
    in_position = False
    
    for i in range(5, len(df)):
        if in_position:
            should_exit, reason = get_exit_signal(df, i, current_trend)
            if should_exit:
                signals.append({
                    'time': df['Datetime'].iloc[i],
                    'action': 'EXIT',
                    'price': df['Close'].iloc[i],
                    'reason': reason
                })
                in_position = False
                current_trend = None
        else:
            # Look for new trend entry
            price_change = ((df['Close'].iloc[i] - df['Close'].iloc[i-5]) / 
                          df['Close'].iloc[i-5] * 100)
            if abs(price_change) >= 1.0:  # Min trend threshold
                current_trend = 'Up' if price_change > 0 else 'Down'
                signals.append({
                    'time': df['Datetime'].iloc[i],
                    'action': 'ENTER ' + current_trend,
                    'price': df['Close'].iloc[i],
                    'reason': f"New {current_trend} trend"
                })
                in_position = True
    
    return pd.DataFrame(signals)

def format_trend_report(trends_df):
    """Format the trends into a readable report."""
    report = "Major Market Trends Analysis\n"
    report += "=" * 50 + "\n\n"
    
    for i, trend in trends_df.iterrows():
        report += f"Trend #{i+1}: {trend['direction']}\n"
        report += f"Start: {trend['start_time'].strftime('%H:%M:%S')} "
        report += f"@ ${trend['start_price']:.2f}\n"
        report += f"End: {trend['end_time'].strftime('%H:%M:%S')} "
        report += f"@ ${trend['end_price']:.2f}\n"
        report += f"Duration: {trend['duration_minutes']:.0f} minutes\n"
        report += f"Price Change: {trend['price_change_pct']:.2f}%\n"
        report += f"High: ${trend['high']:.2f}\n"
        report += f"Low: ${trend['low']:.2f}\n"
        report += "-" * 40 + "\n\n"
    
    return report

def main():
    # Analyze trends
    trends_df = analyze_major_trends(
        'nne_strategy/data/analysis/trends/NNE_trend_data_20241205.csv',
        min_price_change_pct=1.0,
        min_duration_minutes=15,
        volume_threshold_percentile=75
    )
    
    # Generate and save report
    report = format_trend_report(trends_df)
    with open('major_trends_report.txt', 'w') as f:
        f.write(report)
    
    # Save detailed trends data
    trends_df.to_csv('major_trends_data.csv', index=False)
    
    # Add real-time signal analysis
    df = pd.read_csv('nne_strategy/data/analysis/trends/NNE_trend_data_20241205.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    signals_df = format_trade_signals(df)
    signals_df.to_csv('trade_signals.csv', index=False)
    
    # Print summary of signals
    print("\nReal-time Trading Signals Summary:")
    print("=" * 50)
    for _, signal in signals_df.iterrows():
        print(f"{signal['time'].strftime('%H:%M:%S')} - {signal['action']} @ ${signal['price']:.2f}")
        print(f"Reason: {signal['reason']}")
        print("-" * 40)

if __name__ == "__main__":
    main() 