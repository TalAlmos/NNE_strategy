from datetime import datetime
import sys
from pathlib import Path
import pandas as pd
from trend_finder import TrendFinder
from backtesting.position_tracker import PositionTracker
from counter_move_stats import CounterMoveStats

class BacktestRunner:
    def __init__(self, initial_capital=6000.0):
        """Initialize backtest components"""
        self.trend_finder = TrendFinder()
        self.position_tracker = PositionTracker(initial_capital)
        self.counter_move_stats = CounterMoveStats()
        
    def load_data(self, date):
        """Load price data for specified date"""
        try:
            data_path = Path(__file__).parent / "data" / "stock_raw_data" / f"NNE_data_{date}.csv"
            if not data_path.exists():
                raise FileNotFoundError(f"No data file found for date: {date}")
                
            data = pd.read_csv(data_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            print(f"Loaded {len(data)} records for {date}")
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def run_backtest(self, date):
        """Run backtest for specified date"""
        data = self.load_data(date)
        if data is None:
            return None
        
        results = {
            'date': date,
            'trades': [],
            'trends': [],
            'counter_moves': [],
            'initial_capital': self.position_tracker.initial_capital,
            'final_capital': self.position_tracker.initial_capital
        }
        
        # Initialize first trend
        initial_data = data.iloc[:5]
        if initial_data['Close'].iloc[-1] > initial_data['Close'].iloc[0]:
            current_trend = 'UpTrend'
        else:
            current_trend = 'DownTrend'
        
        trend_start_price = initial_data['Close'].iloc[-1]
        
        # Add initial trend to results with datetime
        results['trends'].append({
            'time': initial_data['Datetime'].iloc[-1],  # Use Datetime column
            'trend': current_trend,
            'price': trend_start_price
        })
        
        # Process each bar after initialization
        for i in range(5, len(data)):
            window = data.iloc[max(0, i-5):i+1]
            current_price = window['Close'].iloc[-1]
            current_time = window['Datetime'].iloc[-1]  # Get current datetime
            price_change = current_price - window['Close'].iloc[-2]
            
            # Track counter-moves
            if current_trend == 'UpTrend':
                counter_move = trend_start_price - current_price
            else:
                counter_move = current_price - trend_start_price
            
            # Create trend info dictionary
            trend_info = {
                'trend': current_trend,
                'trend_changed': False,
                'price': current_price
            }
            
            # Check for trend change based on counter-move size
            if abs(counter_move) > self.counter_move_stats.MEDIUM_MOVE:
                # Potential trend change
                if counter_move > 0 and current_trend == 'UpTrend':
                    current_trend = 'DownTrend'
                    trend_start_price = current_price
                    trend_info['trend'] = current_trend
                    trend_info['trend_changed'] = True
                    results['trends'].append({
                        'time': current_time,  # Use current datetime
                        'trend': current_trend,
                        'price': current_price
                    })
                elif counter_move < 0 and current_trend == 'DownTrend':
                    current_trend = 'UpTrend'
                    trend_start_price = current_price
                    trend_info['trend'] = current_trend
                    trend_info['trend_changed'] = True
                    results['trends'].append({
                        'time': current_time,  # Use current datetime
                        'trend': current_trend,
                        'price': current_price
                    })
            
            results['counter_moves'].append({
                'time': window.index[-1],
                'size': counter_move,
                'price': current_price,
                'trend': current_trend
            })
            
            # Update position tracking
            self.position_tracker.update_position(current_price, window.index[-1])
            
            # Check for position exit
            if self.position_tracker.position is not None:
                if self.position_tracker.should_exit(
                    trend_info, 
                    current_price, 
                    current_time,
                    window  # Pass window for volatility calculation
                ):
                    trade_result = self.position_tracker.exit_position(
                        current_price,
                        current_time,
                        reason=f"Trend: {trend_info['trend']}"
                    )
                    if trade_result:
                        results['trades'].append(trade_result)
            
            # Check for position entry
            elif self.position_tracker.can_enter_position(trend_info, current_price, window.index[-1]):
                # Enter position in trend direction
                trade_result = self.position_tracker.enter_position(
                    trend_info['trend'],
                    current_price,
                    current_time  # Use current_time instead of window.index[-1]
                )
                if trade_result:
                    results['trades'].append(trade_result)
        
        # Close any open position at end of day
        if self.position_tracker.position is not None:
            trade_result = self.position_tracker.exit_position(
                data['Close'].iloc[-1],
                data['Datetime'].iloc[-1],  # Use Datetime column
                reason="End of day"
            )
            if trade_result:
                results['trades'].append(trade_result)
        
        # Record final results
        results['final_capital'] = self.position_tracker.current_capital
        
        self._save_results(results)
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results):
        """Save backtest results to log file"""
        log_dir = Path(__file__).parent / "backtesting" / "result_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"backtest_result_log_{results['date']}.txt"
        
        with open(log_file, 'w') as f:
            f.write("NNE Strategy Backtest Results\n")
            f.write("=" * 40 + "\n\n")
            
            # Write summary
            f.write(f"Date: {results['date']}\n")
            f.write(f"Initial Capital: ${results['initial_capital']:,.2f}\n")
            f.write(f"Final Capital: ${results['final_capital']:,.2f}\n")
            pnl = results['final_capital'] - results['initial_capital']
            f.write(f"P/L: ${pnl:,.2f} ({pnl/results['initial_capital']*100:.2f}%)\n\n")
            
            # Write trend changes
            f.write("Trend Changes:\n")
            f.write("-" * 20 + "\n")
            for trend in results['trends']:
                time_str = pd.to_datetime(trend['time']).strftime('%H:%M')
                f.write(f"Time: {time_str}\n")
                f.write(f"Trend: {trend['trend']}\n")
                f.write(f"Price: ${trend['price']:.2f}\n\n")
            
            # Write trades with formatted time
            f.write("\nTrades:\n")
            f.write("-" * 20 + "\n")
            for trade in results['trades']:
                time_str = pd.to_datetime(trade['time']).strftime('%H:%M')
                f.write(f"Time: {time_str}\n")
                f.write(f"Action: {trade['action']}\n")
                f.write(f"Price: ${trade['price']:.2f}\n")
                if 'profit_loss' in trade:
                    f.write(f"P/L: ${trade['profit_loss']:.2f}\n")
                f.write("\n")
        
        print(f"\nResults saved to: {log_file}")
    
    def _print_summary(self, results):
        """Print backtest summary"""
        print("\nBacktest Results Summary")
        print("=" * 40)
        print(f"Date: {results['date']}")
        print(f"\nCapital:")
        print(f"Initial: ${results['initial_capital']:,.2f}")
        print(f"Final: ${results['final_capital']:,.2f}")
        pnl = results['final_capital'] - results['initial_capital']
        print(f"P/L: ${pnl:,.2f} ({pnl/results['initial_capital']*100:.2f}%)")
        
        print(f"\nActivity:")
        print(f"Number of Trades: {len(results['trades'])}")
        print(f"Number of Trends: {len(results['trends'])}")

def main():
    # Parse command line arguments
    date = sys.argv[1] if len(sys.argv) > 1 else '20241212'
    initial_capital = float(sys.argv[2]) if len(sys.argv) > 2 else 6000.0
    
    print("\nBacktest Configuration:")
    print("-" * 30)
    print(f"Date: {date}")
    print(f"Initial Capital: ${initial_capital:,.2f}\n")
    
    # Run backtest
    runner = BacktestRunner(initial_capital)
    results = runner.run_backtest(date)
    
    if results:
        print("\nBacktest completed successfully!")
    else:
        print("\nBacktest failed!")

if __name__ == "__main__":
    main() 