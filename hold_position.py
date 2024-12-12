import pandas as pd
import glob
import os

# Define the base directory (NNE_strategy)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_MOVE_DIR = os.path.join(BASE_DIR, "counter_move_analysis")

def create_sample_data():
    """
    Create sample counter-move analysis files for testing
    """
    if not os.path.exists(COUNTER_MOVE_DIR):
        os.makedirs(COUNTER_MOVE_DIR)
        print(f"Created directory: {COUNTER_MOVE_DIR}")
    
    # Create a sample analysis file
    sample_data = """Counter-move in UpTrend
Duration: 5 minutes
Price Change: -0.25
Volume Change: 1.2

Counter-move in UpTrend
Duration: 3 minutes
Price Change: -0.15
Volume Change: 0.8

Counter-move in DownTrend
Duration: 4 minutes
Price Change: 0.20
Volume Change: 1.1
"""
    
    sample_file = os.path.join(COUNTER_MOVE_DIR, "detailed_countermove_analysis_sample.txt")
    with open(sample_file, 'w') as f:
        f.write(sample_data)
    print(f"Created sample data file: {sample_file}")

class CounterMoveStats:
    def __init__(self, data_directory=COUNTER_MOVE_DIR):
        """
        Initialize counter-move statistics from historical data
        """
        self.stats = self._load_historical_data(data_directory)
        
    def _load_historical_data(self, directory):
        """
        Load and process all historical counter-move analysis files from existing logs
        """
        all_stats = {
            'UpTrend': {
                'durations': [],
                'price_changes': [],
                'volume_changes': [],
                'success_rate': 0
            },
            'DownTrend': {
                'durations': [],
                'price_changes': [],
                'volume_changes': [],
                'success_rate': 0
            }
        }
        
        try:
            # Find all analysis files
            pattern = os.path.join(directory, "*.txt")
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: No historical data files found in {directory}")
                return all_stats
            
            print(f"Found {len(files)} historical data files in {directory}")
            
            for file in files:
                print(f"Processing file: {os.path.basename(file)}")
                with open(file, 'r') as f:
                    lines = f.readlines()
                    current_trend = None
                    
                    for line in lines:
                        line = line.strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                            
                        # Identify trend type
                        if "Counter-move in" in line:
                            if "UpTrend" in line:
                                current_trend = 'UpTrend'
                            elif "DownTrend" in line:
                                current_trend = 'DownTrend'
                            continue
                        
                        # Only process data if we have a valid trend
                        if current_trend is not None:
                            try:
                                if "Duration:" in line:
                                    duration = float(line.split(":")[1].strip().split()[0])
                                    all_stats[current_trend]['durations'].append(duration)
                                elif "Price Change:" in line:
                                    price_change = float(line.split(":")[1].strip().split()[0])
                                    all_stats[current_trend]['price_changes'].append(abs(price_change))
                                elif "Volume Change:" in line:
                                    volume_change = float(line.split(":")[1].strip().split()[0])
                                    all_stats[current_trend]['volume_changes'].append(volume_change)
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Error processing line in {file}: {line}")
                                print(f"Error: {e}")
                                continue
                        
                        # Reset trend at the end of a counter-move section
                        if line == "----------------------------------------":
                            current_trend = None
            
            # Calculate statistics if we have data
            for trend_type in ['UpTrend', 'DownTrend']:
                if all_stats[trend_type]['durations']:
                    all_stats[trend_type]['avg_duration'] = sum(all_stats[trend_type]['durations']) / len(all_stats[trend_type]['durations'])
                    all_stats[trend_type]['max_duration'] = max(all_stats[trend_type]['durations'])
                    all_stats[trend_type]['avg_price_change'] = sum(all_stats[trend_type]['price_changes']) / len(all_stats[trend_type]['price_changes'])
                    all_stats[trend_type]['max_price_change'] = max(all_stats[trend_type]['price_changes'])
                else:
                    print(f"Warning: No data found for {trend_type}")
                    all_stats[trend_type].update({
                        'avg_duration': 0,
                        'max_duration': 0,
                        'avg_price_change': 0,
                        'max_price_change': 0
                    })
            
        except Exception as e:
            print(f"Error processing historical data: {e}")
            return all_stats
        
        return all_stats

def evaluate_position(current_price, current_trend, counter_move_duration, counter_move_price_change, stats):
    """
    Evaluate whether to hold or exit a position based on historical statistics
    
    Args:
        current_price (float): Current stock price
        current_trend (str): Current trend direction ('UpTrend' or 'DownTrend')
        counter_move_duration (float): Duration of current counter-move
        counter_move_price_change (float): Price change in current counter-move
        stats (CounterMoveStats): Historical counter-move statistics
        
    Returns:
        dict: Decision and confidence level
    """
    trend_stats = stats.stats[current_trend]
    
    # Calculate how current counter-move compares to historical data
    duration_percentile = sum(d < counter_move_duration for d in trend_stats['durations']) / len(trend_stats['durations'])
    price_change_percentile = sum(p < abs(counter_move_price_change) for p in trend_stats['price_changes']) / len(trend_stats['price_changes'])
    
    # Decision making logic
    hold_confidence = 1.0
    
    # Reduce confidence based on duration
    if duration_percentile > 0.8:  # Counter-move longer than 80% of historical moves
        hold_confidence *= 0.5
    elif duration_percentile > 0.6:
        hold_confidence *= 0.8
    
    # Reduce confidence based on price change
    if price_change_percentile > 0.8:  # Larger price change than 80% of historical moves
        hold_confidence *= 0.4
    elif price_change_percentile > 0.6:
        hold_confidence *= 0.7
    
    decision = {
        'action': 'HOLD' if hold_confidence > 0.5 else 'EXIT',
        'confidence': hold_confidence,
        'analysis': {
            'duration_percentile': duration_percentile,
            'price_change_percentile': price_change_percentile,
            'avg_historical_duration': trend_stats['avg_duration'],
            'avg_historical_price_change': trend_stats['avg_price_change']
        }
    }
    
    return decision

def monitor_position(ticker, position_type, entry_price, current_price, current_trend):
    """
    Monitor and evaluate an open position
    
    Args:
        ticker (str): Stock ticker
        position_type (str): 'LONG' or 'SHORT'
        entry_price (float): Position entry price
        current_price (float): Current stock price
        current_trend (str): Current trend direction
    
    Returns:
        dict: Position evaluation and recommendation
    """
    # Load historical statistics
    stats = CounterMoveStats()
    
    # Calculate current position metrics
    duration = 0  # This should be calculated from position entry time
    price_change = current_price - entry_price
    
    # Evaluate position
    evaluation = evaluate_position(
        current_price=current_price,
        current_trend=current_trend,
        counter_move_duration=duration,
        counter_move_price_change=price_change,
        stats=stats
    )
    
    return {
        'ticker': ticker,
        'position_type': position_type,
        'entry_price': entry_price,
        'current_price': current_price,
        'profit_loss': price_change if position_type == 'LONG' else -price_change,
        'recommendation': evaluation
    }

# Example usage
if __name__ == "__main__":
    # Example monitoring of a position
    result = monitor_position(
        ticker="NNE",
        position_type="LONG",
        entry_price=25.50,
        current_price=26.00,
        current_trend="UpTrend"
    )
    
    print("\nPosition Analysis:")
    print("=" * 50)
    print(f"Ticker: {result['ticker']}")
    print(f"Position: {result['position_type']}")
    print(f"P/L: ${result['profit_loss']:.2f}")
    print(f"\nRecommendation: {result['recommendation']['action']}")
    print(f"Confidence: {result['recommendation']['confidence']:.2f}")
    print("\nAnalysis Details:")
    for key, value in result['recommendation']['analysis'].items():
        print(f"{key}: {value:.2f}") 