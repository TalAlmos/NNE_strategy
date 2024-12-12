import yfinance as yf
from datetime import datetime, timedelta
import time
from counter_move_stats import CounterMoveStats

class YFinanceMonitor:
    def __init__(self, ticker, position_type, entry_price, current_trend):
        """
        Initialize YFinance position monitor
        
        Args:
            ticker (str): Stock ticker symbol
            position_type (str): 'LONG' or 'SHORT'
            entry_price (float): Position entry price
            current_trend (str): Current trend direction ('UpTrend' or 'DownTrend')
        """
        self.ticker = ticker
        self.position_type = position_type
        self.entry_price = entry_price
        self.current_trend = current_trend
        self.stats = CounterMoveStats()
        self.trend_start_time = datetime.now()
        self.last_price = entry_price
        
    def get_live_data(self, interval='1m'):
        """
        Get real-time stock data from yfinance
        
        Returns:
            dict: Latest stock data including price and timestamp
        """
        try:
            # Get live data
            stock = yf.Ticker(self.ticker)
            live_data = stock.history(period='1d', interval=interval)
            
            if live_data.empty:
                raise ValueError("No live data available")
                
            # Get latest data point
            latest = live_data.iloc[-1]
            return {
                'timestamp': live_data.index[-1],
                'price': latest['Close'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'open': latest['Open']
            }
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return None
            
    def calculate_counter_move(self, current_price):
        """
        Calculate counter-move metrics from current position
        """
        price_change = current_price - self.last_price
        duration = (datetime.now() - self.trend_start_time).total_seconds() / 60  # in minutes
        
        # Determine if this is a counter-move
        is_counter_move = (
            (self.current_trend == 'UpTrend' and price_change < 0) or
            (self.current_trend == 'DownTrend' and price_change > 0)
        )
        
        return {
            'is_counter_move': is_counter_move,
            'duration': duration,
            'price_change': abs(price_change)
        }
    
    def evaluate_position(self, current_price):
        """
        Evaluate current position based on counter-move analysis
        """
        counter_move = self.calculate_counter_move(current_price)
        trend_stats = self.stats.get_trend_stats(self.current_trend)
        
        # Calculate position metrics
        profit_loss = current_price - self.entry_price
        if self.position_type == 'SHORT':
            profit_loss = -profit_loss
            
        # Get historical statistics
        avg_duration = trend_stats.get('avg_duration', 0)
        avg_price_change = trend_stats.get('avg_price_change', 0)
        
        # Decision making logic
        hold_confidence = 1.0
        
        if counter_move['is_counter_move']:
            # Compare with historical counter-moves
            if counter_move['duration'] > avg_duration:
                hold_confidence *= 0.7
            if counter_move['price_change'] > avg_price_change:
                hold_confidence *= 0.6
                
            # Check percentiles if available
            if 'duration_percentiles' in trend_stats:
                if counter_move['duration'] > trend_stats['duration_percentiles']['75']:
                    hold_confidence *= 0.5
                    
            if 'price_change_percentiles' in trend_stats:
                if counter_move['price_change'] > trend_stats['price_change_percentiles']['75']:
                    hold_confidence *= 0.4
        
        return {
            'action': 'HOLD' if hold_confidence > 0.5 else 'EXIT',
            'confidence': hold_confidence,
            'profit_loss': profit_loss,
            'counter_move': counter_move
        }
    
    def monitor_position(self, update_interval=60):
        """
        Continuously monitor position with live data feed
        
        Args:
            update_interval (int): Seconds between updates
        """
        print(f"\nStarting position monitor for {self.ticker}")
        print(f"Position Type: {self.position_type}")
        print(f"Entry Price: ${self.entry_price:.2f}")
        print(f"Current Trend: {self.current_trend}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 50)
        
        try:
            while True:
                # Get live data
                live_data = self.get_live_data()
                
                if live_data:
                    current_price = live_data['price']
                    
                    # Evaluate position
                    evaluation = self.evaluate_position(current_price)
                    
                    # Update last price
                    self.last_price = current_price
                    
                    # Display real-time analysis
                    print(f"\nTime: {live_data['timestamp'].strftime('%H:%M:%S')}")
                    print(f"Current Price: ${current_price:.2f}")
                    print(f"P/L: ${evaluation['profit_loss']:.2f}")
                    print(f"Recommendation: {evaluation['action']}")
                    print(f"Confidence: {evaluation['confidence']:.2f}")
                    
                    if evaluation['counter_move']['is_counter_move']:
                        print("\nCounter-Move Detected:")
                        print(f"Duration: {evaluation['counter_move']['duration']:.1f} minutes")
                        print(f"Price Change: ${evaluation['counter_move']['price_change']:.2f}")
                    
                    # Alert on significant changes
                    if evaluation['action'] == 'EXIT' and evaluation['confidence'] < 0.4:
                        print("\n!!! ALERT: Consider closing position !!!")
                    
                    print("-" * 50)
                
                # Wait for next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nStopping position monitor...")
        except Exception as e:
            print(f"\nError in position monitor: {e}")

def main():
    """
    Main function to run the position monitor
    """
    # Example usage
    monitor = YFinanceMonitor(
        ticker="NNE",
        position_type="LONG",
        entry_price=25.50,
        current_trend="UpTrend"
    )
    
    # Start monitoring with 1-minute updates
    monitor.monitor_position(update_interval=60)

if __name__ == "__main__":
    main() 