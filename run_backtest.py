from backtesting.backtest_engine import BacktestEngine
from datetime import datetime
import sys

def run_single_day(date, initial_capital=6000.0):
    """
    Run backtest for a single day
    
    Args:
        date (str): Date in YYYYMMDD format
        initial_capital (float): Starting capital
    """
    print(f"\nRunning backtest for {date}")
    print("=" * 50)
    
    engine = BacktestEngine(initial_capital=initial_capital)
    results = engine.run_backtest(date)
    
    return results

def main():
    # Default values
    initial_capital = 6000.0
    
    # Get date from command line or use default
    if len(sys.argv) > 1:
        test_date = sys.argv[1]
    else:
        test_date = '20241209'  # Default to December 9th
        
    # Get initial capital if provided
    if len(sys.argv) > 2:
        try:
            initial_capital = float(sys.argv[2])
        except ValueError:
            print(f"Invalid capital amount: {sys.argv[2]}")
            print("Using default: $6000.0")
    
    print("\nBacktest Configuration:")
    print("-" * 30)
    print(f"Date: {test_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    
    # Run backtest
    results = run_single_day(test_date, initial_capital)
    
    if results:
        print("\nBacktest completed successfully!")
    else:
        print("\nBacktest failed!")

if __name__ == "__main__":
    main() 