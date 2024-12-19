import pandas as pd
import numpy as np
from datetime import datetime
from nne_strategy.trend_analysis import TrendAnalysis
from pathlib import Path
from typing import List, Dict
from countermove_analyzer import CountermoveAnalyzer

class TradingSimulation:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.position = None  # None, 'LONG', or 'SHORT'
        self.position_size = 0
        self.entry_price = 0
        self.trades = []
        self.trend_analyzer = TrendAnalysis()  # Use our improved trend analysis
        self.trailing_high = 0
        self.trailing_low = 0
        self.entry_time = None
        
    def simulate_trading_day(self, data_file: str):
        """Simulate trading for one day."""
        print(f"\nLoading data from {data_file}")
        
        try:
            # Read full day's data
            df = pd.read_csv(data_file)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Extract the date from the filename (format: NNE_data_YYYYMMDD.csv)
            date_str = data_file.split('_')[-1].split('.')[0]
            target_date = pd.to_datetime(date_str)
            
            # Filter data for the target date
            df = df[df['Datetime'].dt.date == target_date.date()]
            
            # Ensure timezone is America/New_York
            if df['Datetime'].dt.tz is None:
                df['Datetime'] = df['Datetime'].dt.tz_localize('America/New_York')
            else:
                df['Datetime'] = df['Datetime'].dt.tz_convert('America/New_York')
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for NaN values
                if df[col].isna().any():
                    print(f"WARNING: Found NaN values in {col} column")
                    df = df.dropna(subset=[col])
            
            # Sort by datetime to ensure chronological order
            df = df.sort_values('Datetime')
            df = df.reset_index(drop=True)
            
            # Validate data
            print(f"\nValidating data for {target_date.strftime('%Y-%m-%d')}")
            print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            print(f"First price: ${df.iloc[0]['Close']:.2f}")
            print(f"Last price: ${df.iloc[-1]['Close']:.2f}")
            print(f"Total minutes: {len(df)}")
            
            # Add detailed price validation for specific times
            for time_check in ['14:50']:
                target_time = pd.Timestamp(f"{target_date.date()} {time_check}", tz='America/New_York')
                target_data = df[df['Datetime'] == target_time]
                if not target_data.empty:
                    price = target_data.iloc[0]['Close']
                    print(f"\nPrice at {time_check} on {target_date.strftime('%Y-%m-%d')}: ${price:.2f}")
                    print(f"Raw data at {time_check}:")
                    print(target_data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string())
                    
                    # For December 16th, verify the specific price
                    if target_date.date().strftime('%Y-%m-%d') == '2024-12-16':
                        if time_check == '14:50':
                            expected_price = 25.93
                            if abs(price - expected_price) > 0.01:
                                print(f"WARNING: Price mismatch at {time_check}!")
                                print(f"Expected: ${expected_price:.2f}, Found: ${price:.2f}")
                            else:
                                print(f"✓ Price at {time_check} matches expected value of ${expected_price:.2f}")
            
            if df['Close'].max() > df['Close'].mean() * 1.5 or df['Close'].min() < df['Close'].mean() * 0.5:
                print("WARNING: Detected potentially invalid prices!")
                return None
                
            return df
            
        except Exception as e:
            print(f"Error occurred while processing {data_file}: {e}")
            return None
            
    def simulate_multiple_days(self, data_dir: str = "nne_strategy/data/raw"):
        """Run simulation across multiple days."""
        data_files = sorted(Path(data_dir).glob("NNE_data_*.csv"))
        print(f"\nStarting multi-day simulation with ${self.capital:.2f}")
        print(f"Found {len(data_files)} days to simulate")
        print("=" * 50)
        
        all_trades = []
        daily_results = []
        
        for file in data_files:
            date = file.stem.split('_')[-1]
            print(f"\nSimulating {date}...")
            
            # Store initial capital for this day
            start_capital = self.capital
            
            # Load and validate data for this day
            df = self.simulate_trading_day(str(file))
            if df is None:
                print(f"Skipping {date} due to data loading error")
                continue
                
            # Process the day's data
            self._process_day_data(df)
            
            # Calculate daily metrics
            daily_profit = self.capital - start_capital
            daily_trades = len(self.trades) - len(all_trades)
            
            # Store daily results
            daily_results.append({
                'date': date,
                'profit': daily_profit,
                'trades': daily_trades,
                'win_rate': sum(1 for t in self.trades[len(all_trades):] if t['profit'] > 0) / daily_trades if daily_trades > 0 else 0,
                'largest_win': max([t['profit'] for t in self.trades[len(all_trades):]], default=0),
                'largest_loss': min([t['profit'] for t in self.trades[len(all_trades):]], default=0)
            })
            
            # Add this day's trades to all trades
            all_trades.extend(self.trades[len(all_trades):])
        
        # Generate performance report
        self._generate_performance_report(daily_results, all_trades)
        
    def _process_day_data(self, df: pd.DataFrame):
        """Process a single day's data."""
        current_data = []
        position_profit = 0
        
        # Process minute by minute
        for i in range(len(df)):
            current_minute = df.iloc[i].copy()
            current_data.append(current_minute)
            
            # Log every 30th price for verification
            if i % 30 == 0:
                print(f"Price at {current_minute['Datetime']}: ${current_minute['Close']:.2f}")
            
            # Skip first 30 minutes to gather initial data
            if i < 30:
                continue
            
            # Create DataFrame of available data up to this point
            available_data = pd.DataFrame(current_data).copy()
            
            # Make trading decision
            self._make_trading_decision(available_data, current_minute)
            
            # Update position profit if in a trade
            if self.position:
                current_price = current_minute['Close']
                if self.position == 'LONG':
                    position_profit = (current_price - self.entry_price) * self.position_size
                else:  # SHORT
                    position_profit = (self.entry_price - current_price) * self.position_size
        
        # Close any open position at end of day
        if self.position:
            self._close_position(df.iloc[-1]['Close'], "End of Day")
    
    def _make_trading_decision(self, available_data: pd.DataFrame, current_minute: pd.Series):
        """Make trading decision based on available data."""
        # Ensure we have the correct current price
        current_price = float(current_minute['Close'])
        current_time = pd.to_datetime(current_minute['Datetime'])
        
        # Skip trading during the first 15 minutes (warm-up period)
        market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        warm_up_end = market_open_time + pd.Timedelta(minutes=15)
        if current_time <= warm_up_end:
            return
            
        # Initialize countermove analyzer if not exists
        if not hasattr(self, 'countermove_analyzer'):
            self.countermove_analyzer = CountermoveAnalyzer()
            self.countermove_analyzer.analyze_historical_countermoves(available_data)
        
        # Analyze trends using our improved trend analysis
        trend_data = self.trend_analyzer.identify_trends(available_data)
        current_row = trend_data.iloc[-1]
        
        # Calculate ATR for dynamic stops
        atr = self._calculate_atr(available_data.tail(30))
        
        # If we have a position, check for risk management first
        if self.position:
            # Calculate current P/L
            if self.position == 'LONG':
                current_pl = (current_price - self.entry_price) * self.position_size
            else:  # SHORT
                current_pl = (self.entry_price - current_price) * self.position_size
            
            # Hard stop loss at 2% of position value
            position_value = self.entry_price * self.position_size
            if current_pl < -(position_value * 0.02):
                self._close_position(current_price, "Hard Stop Loss Hit")
                return
            
            # Check for countermove or reversal
            if current_row['IsReversal']:
                self._close_position(current_price, "Reversal Detected")
                return
            elif current_row['IsCountermove']:
                # Tighten stops on countermove
                if self.position == 'LONG':
                    self.trailing_high = max(self.trailing_high, current_price)
                    tight_stop = self.trailing_high - (1.5 * atr)  # Tighter stop
                    if current_price < tight_stop:
                        self._close_position(current_price, "Tightened Stop Loss Hit")
                        return
                else:  # SHORT
                    self.trailing_low = min(self.trailing_low, current_price)
                    tight_stop = self.trailing_low + (1.5 * atr)  # Tighter stop
                    if current_price > tight_stop:
                        self._close_position(current_price, "Tightened Stop Loss Hit")
                        return
            
        # If we have no position, check for entry
        if not self.position:
            # Check for strong trend with entry signal
            if (current_row['Trend'] == 'Up' and 
                current_row['EntrySignal'] and 
                current_row['TrendStrength'] > 0.6 and 
                not current_row['IsCountermove']):
                self._enter_long(current_price, current_time)
            elif (current_row['Trend'] == 'Down' and 
                  current_row['EntrySignal'] and 
                  current_row['TrendStrength'] > 0.6 and 
                  not current_row['IsCountermove']):
                self._enter_short(current_price, current_time)
        
        # If we have a position, check for exit
        elif self.position == 'LONG':
            # Check minimum hold time (5 minutes)
            if (current_time - self.entry_time) < pd.Timedelta(minutes=5):
                return
                
            # Dynamic trailing stop based on ATR and trend strength
            trend_strength = current_row['TrendStrength']
            trailing_stop = max(
                self.trailing_high - (3.0 * atr / trend_strength),
                self.entry_price - (2.0 * atr)
            )
            
            # Update trailing high if price is higher
            self.trailing_high = max(self.trailing_high, current_price)
            
            # Exit conditions for long position
            if current_row['ExitSignal'] or current_price <= trailing_stop:
                self._close_position(current_price, "Strong Counter-Signal/Trailing Stop")
                
        elif self.position == 'SHORT':
            # Check minimum hold time (5 minutes)
            if (current_time - self.entry_time) < pd.Timedelta(minutes=5):
                return
                
            # Dynamic trailing stop based on ATR and trend strength
            trend_strength = current_row['TrendStrength']
            trailing_stop = min(
                self.trailing_low + (3.0 * atr / trend_strength),
                self.entry_price + (2.0 * atr)
            )
            
            # Update trailing low if price is lower
            self.trailing_low = min(self.trailing_low, current_price)
            
            # Exit conditions for short position
            if current_row['ExitSignal'] or current_price >= trailing_stop:
                self._close_position(current_price, "Strong Counter-Signal/Trailing Stop")
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < 2:
            return 0.0
            
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else tr.mean()
    
    def _enter_long(self, price: float, time: datetime):
        """Enter a long position."""
        self.position = 'LONG'
        self.position_size = int(self.capital * 0.95 / price)
        self.entry_price = price
        self.entry_time = time  # Track entry time
        self.trailing_high = price  # Initialize trailing high
        print(f"\n{time.strftime('%H:%M:%S')} - Entering LONG")
        print(f"Price: ${price:.2f}")
        print(f"Size: {self.position_size} shares")
    
    def _enter_short(self, price: float, time: datetime):
        """Enter a short position."""
        self.position = 'SHORT'
        self.position_size = int(self.capital * 0.95 / price)
        self.entry_price = price
        self.entry_time = time  # Track entry time
        self.trailing_low = price  # Initialize trailing low
        print(f"\n{time.strftime('%H:%M:%S')} - Entering SHORT")
        print(f"Price: ${price:.2f}")
        print(f"Size: {self.position_size} shares")
    
    def _close_position(self, price: float, reason: str):
        """Close current position."""
        if self.position == 'LONG':
            profit = (price - self.entry_price) * self.position_size
        else:  # SHORT
            profit = (self.entry_price - price) * self.position_size
        
        self.capital += profit
        
        # Record trade
        self.trades.append({
            'type': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'size': self.position_size,
            'profit': profit,
            'exit_time': datetime.now(),
            'duration': len(self.trades) + 1,
            'reason': reason
        })
        
        print(f"\nClosing {self.position} position")
        print(f"Exit Price: ${price:.2f}")
        print(f"Profit/Loss: ${profit:.2f}")
        print(f"Reason: {reason}")
        
        # Reset position
        self.position = None
        self.position_size = 0
        self.entry_price = 0
    
    def _generate_performance_report(self, daily_results: List[Dict], all_trades: List[Dict]) -> None:
        """Generate comprehensive performance report."""
        print("\nTrading Simulation Performance Report")
        print("=" * 50)
        
        # Overall Performance
        total_profit = sum(day['profit'] for day in daily_results)
        total_trades = sum(day['trades'] for day in daily_results)
        profitable_trades = sum(1 for trade in all_trades if trade['profit'] > 0)
        
        print("\nOverall Performance:")
        print(f"Starting Capital: ${10000:.2f}")
        print(f"Ending Capital: ${self.capital:.2f}")
        print(f"Total Profit/Loss: ${total_profit:.2f}")
        print(f"Return on Investment: {(total_profit/10000)*100:.1f}%")
        print(f"Total Trading Days: {len(daily_results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {(profitable_trades/total_trades)*100:.1f}%")
        
        # Daily Statistics
        profitable_days = sum(1 for day in daily_results if day['profit'] > 0)
        avg_daily_profit = np.mean([day['profit'] for day in daily_results])
        avg_trades_per_day = np.mean([day['trades'] for day in daily_results])
        
        print("\nDaily Statistics:")
        print(f"Profitable Days: {profitable_days}/{len(daily_results)} ({profitable_days/len(daily_results)*100:.1f}%)")
        print(f"Average Daily Profit: ${avg_daily_profit:.2f}")
        print(f"Average Trades per Day: {avg_trades_per_day:.1f}")
        
        # Trade Statistics
        if all_trades:
            profits = [trade['profit'] for trade in all_trades]
            print("\nTrade Statistics:")
            print(f"Largest Win: ${max(profits):.2f}")
            print(f"Largest Loss: ${min(profits):.2f}")
            print(f"Average Win: ${np.mean([p for p in profits if p > 0]):.2f}")
            print(f"Average Loss: ${abs(np.mean([p for p in profits if p < 0])):.2f}")
            
            # Position Type Analysis
            long_trades = [t for t in all_trades if t['type'] == 'LONG']
            short_trades = [t for t in all_trades if t['type'] == 'SHORT']
            
            print("\nPosition Analysis:")
            if long_trades:
                long_profit = sum(t['profit'] for t in long_trades)
                long_win_rate = sum(1 for t in long_trades if t['profit'] > 0) / len(long_trades)
                print(f"LONG - Count: {len(long_trades)}, Profit: ${long_profit:.2f}, Win Rate: {long_win_rate*100:.1f}%")
            
            if short_trades:
                short_profit = sum(t['profit'] for t in short_trades)
                short_win_rate = sum(1 for t in short_trades if t['profit'] > 0) / len(short_trades)
                print(f"SHORT - Count: {len(short_trades)}, Profit: ${short_profit:.2f}, Win Rate: {short_win_rate*100:.1f}%")
        
        # Save detailed results to file
        self._save_detailed_results(daily_results, all_trades)
    
    def _save_detailed_results(self, daily_results: List[Dict], all_trades: List[Dict]) -> None:
        """Save detailed trading results to CSV files."""
        # Save daily results
        daily_df = pd.DataFrame(daily_results)
        daily_df.to_csv('daily_trading_results.csv', index=False)
        
        # Save trade details
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv('trade_details.csv', index=False)
        
        print("\nDetailed results saved to:")
        print("- daily_trading_results.csv")
        print("- trade_details.csv")

def main():
    # Run simulation on all available days
    simulator = TradingSimulation(initial_capital=10000)
    simulator.simulate_multiple_days()

if __name__ == "__main__":
    main() 