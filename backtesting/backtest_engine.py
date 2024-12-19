from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))
from nne_strategy.counter_move_stats import CounterMoveStats
from nne_strategy.trend_analysis import TrendAnalysis
from nne_strategy.backtesting.position_tracker import PositionTracker
from backtesting.risk_manager import RiskManager, RiskConfig

class BacktestEngine:
    def __init__(self, initial_capital: float = 6000.0):
        """Initialize backtesting engine with risk management"""
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
            
        self.initial_capital = initial_capital
        self.position_tracker = PositionTracker(initial_capital)
        self.trend_analyzer = TrendAnalysis()
        
        # Initialize data tracking
        self.data = None
        self.current_trend = None
        self.trend_start_price = 0
        self.trend_start_time = None
        self.high_since_start = float('-inf')
        self.low_since_start = float('inf')
        self.sma5_buffer = []
        
        # Load statistical parameters
        self.stats = CounterMoveStats()
        self.uptrend_stats = self.stats.get_trend_stats('UpTrend')
        self.downtrend_stats = self.stats.get_trend_stats('DownTrend')
        
        # Trading parameters
        self.ENTRY_THRESHOLD = 0.003  # Enter on 0.3% counter-move
        self.MIN_TIME_BETWEEN_TRADES = pd.Timedelta(minutes=5)
        self.last_trade_time = None
        
    def load_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load raw data for specified date"""
        try:
            script_dir = Path(__file__).parent.parent
            data_path = script_dir / "data" / "stock_raw_data" / f"NNE_data_{date}.csv"
            
            if not data_path.exists():
                raise FileNotFoundError(f"No data file found for date: {date}")
            
            self.data = pd.read_csv(data_path)
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
            print(f"Loaded {len(self.data)} records for {date}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def update_trend(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Update trend status based on new data"""
        # Add validation for required columns
        required_columns = ['Datetime', 'Close', 'High', 'Low']
        if not all(col in row.index for col in required_columns):
            raise ValueError(f"Row missing required columns: {required_columns}")
            
        try:
            # Update SMA5
            self.sma5_buffer.append(row['Close'])
            if len(self.sma5_buffer) > 5:
                self.sma5_buffer.pop(0)
            current_sma = sum(self.sma5_buffer) / len(self.sma5_buffer)
            
            # Update trend extremes
            self.high_since_start = max(self.high_since_start, row['High'])
            self.low_since_start = min(self.low_since_start, row['Low'])
            
            trend_info = {
                'previous_trend': self.current_trend,
                'new_trend': self.current_trend,
                'trend_changed': False,
                'price': row['Close'],
                'time': row['Datetime']
            }
            
            # Check for trend reversal
            if self._check_trend_reversal(row, current_sma):
                self._handle_trend_change(row, trend_info)
            
            return trend_info
            
        except Exception as e:
            print(f"Error updating trend: {e}")
            return None
            
    def detect_counter_move(self, row: pd.Series) -> Optional[Dict]:
        """Detect counter-move with enhanced sensitivity"""
        try:
            recent_high = max(self.sma5_buffer) if self.sma5_buffer else row['Close']
            recent_low = min(self.sma5_buffer) if self.sma5_buffer else row['Close']
            
            counter_move = None
            if self.current_trend == 'UpTrend':
                counter_move_size = (recent_high - row['Close']) / recent_high
                if counter_move_size > self.ENTRY_THRESHOLD:
                    counter_move = self._create_counter_move(row, counter_move_size)
                    
            else:  # DownTrend
                counter_move_size = (row['Close'] - recent_low) / recent_low
                if counter_move_size > self.ENTRY_THRESHOLD:
                    counter_move = self._create_counter_move(row, counter_move_size)
            
            return counter_move
            
        except Exception as e:
            print(f"Error detecting counter-move: {e}")
            return None
            
    def run_backtest(self, date: str) -> Optional[Dict]:
        """Run backtest with enhanced risk management"""
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
            
            # Process each row
            lookback_window = 20
            for i in range(lookback_window, len(self.data)):
                current_row = self.data.iloc[i]
                window = self.data.iloc[i-lookback_window:i+1]
                
                # Update trend
                trend_info = self.update_trend(current_row)
                if trend_info and trend_info['trend_changed']:
                    results['trends'].append(trend_info)
                
                # Detect counter-moves
                counter_move = self.detect_counter_move(current_row)
                if counter_move:
                    results['counter_moves'].append(counter_move)
                    
                    # Check for trade actions
                    if self.position_tracker.position is None:
                        if self._should_enter_trade(counter_move, window):
                            trade = self.position_tracker.enter_position(
                                trend_type=self.current_trend,
                                price=current_row['Close'],
                                time=current_row['Datetime'],
                                window=window
                            )
                            if trade:
                                results['trades'].append(trade)
                    
                # Update existing position
                if self.position_tracker.position is not None:
                    trade = self.position_tracker.update_position(
                        current_row['Close'],
                        current_row['Datetime']
                    )
                    if trade:
                        results['trades'].append(trade)
            
            # Close any open position
            if self.position_tracker.position:
                trade = self.position_tracker.exit_position(
                    self.data.iloc[-1]['Close'],
                    self.data.iloc[-1]['Datetime'],
                    "End of Day"
                )
                if trade:
                    results['trades'].append(trade)
            
            results['final_capital'] = self.position_tracker.current_capital
            self._print_backtest_summary(results)
            return results
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return None
            
    def _check_trend_reversal(self, row: pd.Series, current_sma: float) -> bool:
        """Check for trend reversal conditions"""
        SWING_THRESHOLD = 0.50
        
        if self.current_trend == 'UpTrend':
            return (row['Low'] < (self.high_since_start - SWING_THRESHOLD) and 
                   row['Close'] < current_sma)
        else:  # DownTrend
            return (row['High'] > (self.low_since_start + SWING_THRESHOLD) and 
                   row['Close'] > current_sma)
                   
    def _handle_trend_change(self, row: pd.Series, trend_info: Dict):
        """Handle trend change updates"""
        trend_info['new_trend'] = 'DownTrend' if self.current_trend == 'UpTrend' else 'UpTrend'
        trend_info['trend_changed'] = True
        
        self.current_trend = trend_info['new_trend']
        self.trend_start_price = row['Close']
        self.trend_start_time = row['Datetime']
        self.high_since_start = row['High']
        self.low_since_start = row['Low']
        
    def _create_counter_move(self, row: pd.Series, size: float) -> Dict:
        """Create counter-move information dictionary"""
        return {
            'time': row['Datetime'],
            'size': size,
            'price_range': row['High'] - row['Low'],
            'current_price': row['Close'],
            'trend_direction': self.current_trend
        }
        
    def _should_enter_trade(self, counter_move: Dict, window: pd.DataFrame) -> bool:
        """Enhanced trade entry decision"""
        # Check time between trades
        if self.last_trade_time is not None:
            time_since_last = counter_move['time'] - self.last_trade_time
            if time_since_last < self.MIN_TIME_BETWEEN_TRADES:
                return False
        
        # Calculate trend duration
        trend_duration = (counter_move['time'] - self.trend_start_time).total_seconds() / 60
        
        # Get trend statistics
        stats = self.uptrend_stats if self.current_trend == 'UpTrend' else self.downtrend_stats
        
        # Check trend maturity
        if trend_duration > stats['duration_percentiles']['75']:
            return False
            
        return True
        
    def _print_backtest_summary(self, results: Dict):
        """Print detailed backtest results summary"""
        print("\nBacktest Results Summary")
        print("=" * 40)
        print(f"Date: {results['date']}")
        print(f"\nCapital:")
        print(f"Initial: ${results['initial_capital']:,.2f}")
        print(f"Final: ${results['final_capital']:,.2f}")
        pnl = results['final_capital'] - results['initial_capital']
        print(f"P/L: ${pnl:,.2f} ({pnl/results['initial_capital']*100:.2f}%)")
        
        if results['trades']:
            self._print_trade_statistics(results['trades'])
            
    def _print_trade_statistics(self, trades: List[Dict]):
        """Print detailed trade statistics"""
        profits = [t['profit_loss'] for t in trades if 'profit_loss' in t]
        if profits:
            print(f"\nTrade Performance:")
            print(f"Number of Trades: {len(trades)}")
            print(f"Average P/L: ${sum(profits)/len(profits):,.2f}")
            print(f"Max Profit: ${max(profits):,.2f}")
            print(f"Max Loss: ${min(profits):,.2f}")
            win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
            print(f"Win Rate: {win_rate:.1f}%")