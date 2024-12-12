from datetime import datetime
import pandas as pd
from pathlib import Path
from counter_move_stats import CounterMoveStats

class BacktestEngine:
    def __init__(self, initial_capital=6000.0):
        """Initialize backtesting engine with statistical insights"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        self.position_entry_time = None
        self.current_trend = None
        self.trend_start_price = 0
        self.trend_start_time = None
        self.high_since_start = float('-inf')
        self.low_since_start = float('inf')
        self.trade_history = []
        self.sma5_buffer = []
        
        # Load statistical parameters
        self.stats = CounterMoveStats()
        self.uptrend_stats = self.stats.get_trend_stats('UpTrend')
        self.downtrend_stats = self.stats.get_trend_stats('DownTrend')
        
        # Trading parameters based on statistics
        self.ENTRY_THRESHOLD = 0.04  # Median counter-move size
        self.EXIT_THRESHOLD = 0.07   # 75th percentile counter-move size
        self.STOP_LOSS = 0.05       # Average counter-move size
        self.MIN_PROFIT = 0.02      # 25th percentile counter-move size
        
    def load_data(self, date):
        """
        Load raw data for specified date
        
        Args:
            date (str): Date in YYYYMMDD format
        """
        try:
            script_dir = Path(__file__).parent.parent
            data_path = script_dir / "data" / "stock_raw_data" / f"NNE_data_{date}.csv"
            
            if not data_path.exists():
                raise FileNotFoundError(f"No data file found for date: {date}")
                
            self.data = pd.read_csv(data_path)
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
            print(f"Loaded {len(self.data)} records for {date}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def initialize_trend(self, first_5min_data):
        """
        Initialize trend using first 5 minutes of data
        
        Args:
            first_5min_data (pd.DataFrame): First 5 minutes of price data
        """
        try:
            # Calculate SMA5 from first 5 minutes
            self.sma5_buffer = list(first_5min_data['Close'])
            current_sma = sum(self.sma5_buffer) / len(self.sma5_buffer)
            
            # Get last price of 5-min period
            last_price = first_5min_data.iloc[-1]['Close']
            first_price = first_5min_data.iloc[0]['Open']
            
            # Determine initial trend
            self.current_trend = 'UpTrend' if last_price > current_sma else 'DownTrend'
            self.trend_start_price = first_price
            self.trend_start_time = first_5min_data.iloc[0]['Datetime']
            
            # Initialize trend extremes
            self.high_since_start = first_5min_data['High'].max()
            self.low_since_start = first_5min_data['Low'].min()
            
            print(f"Initial trend determined: {self.current_trend}")
            print(f"Start price: ${self.trend_start_price:.2f}")
            
        except Exception as e:
            print(f"Error initializing trend: {e}")

    def update_trend(self, row):
        """
        Update trend status based on new data
        
        Args:
            row (pd.Series): Current price data row
        Returns:
            dict: Updated trend information
        """
        # Parameters from trend_finder.py
        SWING_THRESHOLD = 0.50
        CONFIRMATION_PERIODS = 3
        VOLUME_FACTOR = 1.5
        
        try:
            # Update SMA5
            self.sma5_buffer.append(row['Close'])
            if len(self.sma5_buffer) > 5:
                self.sma5_buffer.pop(0)
            current_sma = sum(self.sma5_buffer) / len(self.sma5_buffer)
            
            # Update trend extremes
            self.high_since_start = max(self.high_since_start, row['High'])
            self.low_since_start = min(self.low_since_start, row['Low'])
            
            trend_change = False
            trend_info = {
                'previous_trend': self.current_trend,
                'new_trend': self.current_trend,
                'trend_changed': False,
                'price': row['Close'],
                'time': row['Datetime']
            }
            
            # Check for trend reversal
            if self.current_trend == 'UpTrend':
                if row['Low'] < (self.high_since_start - SWING_THRESHOLD):
                    # Confirm downtrend
                    if row['Close'] < current_sma:
                        trend_change = True
                        trend_info['new_trend'] = 'DownTrend'
            else:  # DownTrend
                if row['High'] > (self.low_since_start + SWING_THRESHOLD):
                    # Confirm uptrend
                    if row['Close'] > current_sma:
                        trend_change = True
                        trend_info['new_trend'] = 'UpTrend'
            
            # If trend changed, update tracking variables
            if trend_change:
                self.current_trend = trend_info['new_trend']
                self.trend_start_price = row['Close']
                self.trend_start_time = row['Datetime']
                self.high_since_start = row['High']
                self.low_since_start = row['Low']
                trend_info['trend_changed'] = True
                
                print(f"Trend change detected at {row['Datetime']}")
                print(f"New trend: {self.current_trend}")
                print(f"Price: ${row['Close']:.2f}")
            
            return trend_info
            
        except Exception as e:
            print(f"Error updating trend: {e}")
            return None

    def detect_counter_move(self, row):
        """
        Detect if current price movement is a counter-move
        
        Args:
            row (pd.Series): Current price data row
        Returns:
            dict: Counter-move information if detected, None otherwise
        """
        try:
            price_change = row['Close'] - self.trend_start_price
            
            # Detect counter-move
            is_counter = ((self.current_trend == 'UpTrend' and price_change < 0) or 
                         (self.current_trend == 'DownTrend' and price_change > 0))
            
            if is_counter:
                counter_move = {
                    'time': row['Datetime'],
                    'size': abs(price_change),
                    'price_range': row['High'] - row['Low'],
                    'current_price': row['Close'],
                    'trend_direction': self.current_trend
                }
                return counter_move
                
            return None
            
        except Exception as e:
            print(f"Error detecting counter-move: {e}")
            return None
        
    def execute_trade(self, row, action):
        """
        Execute trade action and update position
        
        Args:
            row (pd.Series): Current price data row
            action (str): Trade action ('ENTER_LONG', 'EXIT_LONG', 'ENTER_SHORT', 'EXIT_SHORT')
        Returns:
            dict: Trade details
        """
        try:
            trade_info = {
                'time': row['Datetime'],
                'price': row['Close'],
                'action': action,
                'position_size': 0,
                'profit_loss': 0
            }
            
            # Calculate position size (number of shares)
            position_value = self.initial_capital * 0.95  # Use 95% of capital
            shares = int(position_value / row['Close'])
            
            if action.startswith('ENTER'):
                if self.position is not None:
                    print(f"Warning: Attempting to enter position while already in {self.position}")
                    return None
                    
                if action == 'ENTER_LONG':
                    self.position = 'LONG'
                    self.position_size = shares
                    self.entry_price = row['Close']
                    self.position_entry_time = row['Datetime']
                    trade_info['position_size'] = shares
                    print(f"\nEntering LONG position:")
                    print(f"Time: {row['Datetime']}")
                    print(f"Price: ${row['Close']:.2f}")
                    print(f"Shares: {shares}")
                    
                elif action == 'ENTER_SHORT':
                    self.position = 'SHORT'
                    self.position_size = shares
                    self.entry_price = row['Close']
                    self.position_entry_time = row['Datetime']
                    trade_info['position_size'] = shares
                    print(f"\nEntering SHORT position:")
                    print(f"Time: {row['Datetime']}")
                    print(f"Price: ${row['Close']:.2f}")
                    print(f"Shares: {shares}")
                    
            elif action.startswith('EXIT'):
                if self.position is None:
                    print("Warning: Attempting to exit non-existent position")
                    return None
                    
                # Calculate profit/loss
                if self.position == 'LONG':
                    profit_loss = (row['Close'] - self.entry_price) * self.position_size
                else:  # SHORT
                    profit_loss = (self.entry_price - row['Close']) * self.position_size
                    
                # Update capital
                self.current_capital += profit_loss
                
                # Record trade details
                trade_info.update({
                    'entry_price': self.entry_price,
                    'exit_price': row['Close'],
                    'position_size': self.position_size,
                    'profit_loss': profit_loss,
                    'position_type': self.position
                })
                
                print(f"\nExiting {self.position} position:")
                print(f"Time: {row['Datetime']}")
                print(f"Entry Price: ${self.entry_price:.2f}")
                print(f"Exit Price: ${row['Close']:.2f}")
                print(f"P/L: ${profit_loss:.2f}")
                print(f"Current Capital: ${self.current_capital:.2f}")
                
                # Reset position
                self.position = None
                self.position_size = 0
                self.entry_price = 0
                
            return trade_info
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None
        
    def should_enter_position(self, counter_move, trend_duration):
        """
        Determine if we should enter a position based on statistics
        
        Args:
            counter_move (dict): Counter-move information
            trend_duration (int): Current trend duration in minutes
        """
        if counter_move['trend_direction'] == 'UpTrend':
            stats = self.uptrend_stats
            # Enter if counter-move size is significant and trend duration is within normal range
            return (counter_move['size'] >= self.ENTRY_THRESHOLD and 
                   trend_duration <= stats['duration_percentiles']['75'])
        else:
            stats = self.downtrend_stats
            return (counter_move['size'] >= self.ENTRY_THRESHOLD and 
                   trend_duration <= stats['duration_percentiles']['75'])
    
    def should_exit_position(self, counter_move, position_duration):
        """
        Determine if we should exit a position
        
        Args:
            counter_move (dict): Counter-move information
            position_duration (int): Current position duration in minutes
        """
        # Exit if counter-move against our position is large
        if counter_move['size'] >= self.EXIT_THRESHOLD:
            return True
            
        # Exit if position duration exceeds typical trend duration
        if self.position == 'LONG':
            if position_duration > self.uptrend_stats['duration_percentiles']['75']:
                return True
        else:
            if position_duration > self.downtrend_stats['duration_percentiles']['75']:
                return True
                
        return False
    
    def run_backtest(self, date):
        """Run backtest with statistical insights"""
        try:
            self.load_data(date)
            if self.data is None:
                return None
            
            results = {
                'date': date,
                'trades': [],
                'trends': [],
                'counter_moves': [],
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital
            }
            
            # Initialize with first 5 minutes
            first_5min = self.data.iloc[:5]
            self.initialize_trend(first_5min)
            trend_start_time = self.data.iloc[5]['Datetime']
            
            # Process each row after first 5 minutes
            for i in range(5, len(self.data)):
                current_row = self.data.iloc[i]
                current_time = current_row['Datetime']
                
                # Update trend status
                trend_info = self.update_trend(current_row)
                if trend_info['trend_changed']:
                    trend_start_time = current_time
                    results['trends'].append(trend_info)
                
                # Calculate trend duration
                trend_duration = (current_time - trend_start_time).total_seconds() / 60
                
                # Detect counter-moves
                counter_move = self.detect_counter_move(current_row)
                if counter_move:
                    results['counter_moves'].append(counter_move)
                    
                    # Trading decisions
                    if self.position is None:
                        if self.should_enter_position(counter_move, trend_duration):
                            action = 'ENTER_LONG' if counter_move['trend_direction'] == 'UpTrend' else 'ENTER_SHORT'
                            trade_result = self.execute_trade(current_row, action)
                            if trade_result:
                                results['trades'].append(trade_result)
                    else:
                        position_duration = (current_time - self.position_entry_time).total_seconds() / 60
                        if self.should_exit_position(counter_move, position_duration):
                            action = f'EXIT_{self.position}'
                            trade_result = self.execute_trade(current_row, action)
                            if trade_result:
                                results['trades'].append(trade_result)
            
            # Close any open position at end of day
            if self.position:
                trade_result = self.execute_trade(self.data.iloc[-1], f'EXIT_{self.position}')
                if trade_result:
                    results['trades'].append(trade_result)
            
            # Calculate final results
            results['final_capital'] = self.current_capital
            results['total_return'] = (self.current_capital - self.initial_capital) / self.initial_capital * 100
            
            self._print_backtest_summary(results)
            return results
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return None 

    def _print_backtest_summary(self, results):
        """Print detailed backtest results summary"""
        print("\nBacktest Results Summary")
        print("=" * 40)
        print(f"Date: {results['date']}")
        print(f"\nCapital:")
        print(f"Initial: ${results['initial_capital']:,.2f}")
        print(f"Final: ${results['final_capital']:,.2f}")
        print(f"Return: {results['total_return']:.2f}%")
        
        print(f"\nActivity:")
        print(f"Number of Trades: {len(results['trades'])}")
        print(f"Number of Trends: {len(results['trends'])}")
        print(f"Number of Counter-Moves: {len(results['counter_moves'])}")
        
        if results['trades']:
            profits = [t['profit_loss'] for t in results['trades'] if 'profit_loss' in t]
            if profits:
                print(f"\nTrade Performance:")
                print(f"Average P/L: ${sum(profits)/len(profits):,.2f}")
                print(f"Max Profit: ${max(profits):,.2f}")
                print(f"Max Loss: ${min(profits):,.2f}")
                win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                print(f"Win Rate: {win_rate:.1f}%")