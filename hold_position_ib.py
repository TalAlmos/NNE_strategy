from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
import time
from counter_move_stats import CounterMoveStats

class IBPositionMonitor(EWrapper, EClient):
    def __init__(self, ticker, position_type, entry_price, current_trend):
        """
        Initialize IB position monitor
        
        Args:
            ticker (str): Stock ticker symbol
            position_type (str): 'LONG' or 'SHORT'
            entry_price (float): Position entry price
            current_trend (str): Current trend direction ('UpTrend' or 'DownTrend')
        """
        EClient.__init__(self, self)
        
        self.ticker = ticker
        self.position_type = position_type
        self.entry_price = entry_price
        self.current_trend = current_trend
        self.stats = CounterMoveStats()
        self.trend_start_time = datetime.now()
        self.last_price = entry_price
        
        # Data storage
        self.current_data = None
        self.is_connected = False
        self.req_id = 1
        
    def error(self, reqId, errorCode, errorString):
        """
        Handle IB API errors
        """
        print(f"Error {errorCode}: {errorString}")
        
    def connectAck(self):
        """
        Callback when connection is established
        """
        self.is_connected = True
        print("Connected to Interactive Brokers")
        
    def tickPrice(self, reqId, tickType, price, attrib):
        """
        Handle real-time price updates
        """
        if tickType == 4:  # Last price
            self.current_data = {
                'timestamp': datetime.now(),
                'price': price
            }
            
    def create_contract(self):
        """
        Create IB contract for the stock
        """
        contract = Contract()
        contract.symbol = self.ticker
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract
        
    def calculate_counter_move(self, current_price):
        """
        Calculate counter-move metrics from current position
        """
        price_change = current_price - self.last_price
        duration = (datetime.now() - self.trend_start_time).total_seconds() / 60
        
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
        
        # Connect to IB
        self.connect('127.0.0.1', 7497, 1)  # Use 7496 for TWS paper trading
        
        # Wait for connection
        while not self.is_connected:
            self.run()
            time.sleep(0.1)
        
        # Request market data
        contract = self.create_contract()
        self.reqMktData(self.req_id, contract, '', False, False, [])
        
        try:
            while True:
                self.run()  # Process IB messages
                
                if self.current_data:
                    current_price = self.current_data['price']
                    
                    # Evaluate position
                    evaluation = self.evaluate_position(current_price)
                    
                    # Update last price
                    self.last_price = current_price
                    
                    # Display real-time analysis
                    print(f"\nTime: {self.current_data['timestamp'].strftime('%H:%M:%S')}")
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
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nStopping position monitor...")
        except Exception as e:
            print(f"\nError in position monitor: {e}")
        finally:
            # Clean up
            self.cancelMktData(self.req_id)
            self.disconnect()

def main():
    """
    Main function to run the position monitor
    """
    # Example usage
    monitor = IBPositionMonitor(
        ticker="NNE",
        position_type="LONG",
        entry_price=25.50,
        current_trend="UpTrend"
    )
    
    # Start monitoring with 1-minute updates
    monitor.monitor_position(update_interval=60)

if __name__ == "__main__":
    main()