import json
import os
import pandas as pd

class BacktestingSystem:
    def __init__(self, data_folder, initial_account_size=100000.0, risk_fraction=0.01):
        """
        Initialize the backtesting system with the consolidated countermove statistics
        """
        self.data_folder = data_folder
        self.account_size = initial_account_size
        self.risk_fraction = risk_fraction

        # Trading state variables
        self.position_open = False
        self.current_trend = None
        self.stop_loss = None
        self.entry_price = None
        self.entry_time = None
        self.trade_history = []

        # Load countermove analysis from the correct location
        stats_file_path = r"D:\NNE_strategy\nne_strategy\data\stats\consolidated_countermove_analysis.json"
        with open(stats_file_path, 'r') as f:
            self.statistics = json.load(f)

        # We'll maintain a "live_data" DF as we receive candles
        self.live_data = None

        # Add tracking variables for real-time analysis
        self.trend_start_price = None
        self.trend_start_time = None
        self.current_move_duration = 0
        self.last_trend_direction = None
        self.min_score_for_entry = 70  # We can adjust this threshold during testing

    def _get_stats_for_day(self, day_date):
        """
        Get statistics for trading decisions
        """
        # No need to process day_date anymore as we use general statistics
        return self.statistics  # Return the entire statistics object

    def set_stop_loss(self, entry_price: float, trend_type: str, stats: dict):
        """
        Set stop-loss using the statistical structure
        """
        try:
            if trend_type == "UpTrend":
                # For bullish trades, use downtrend countermoves stats
                if 'downtrend_countermoves' in stats and 'PriceChange' in stats['downtrend_countermoves']:
                    avg_price_change = abs(stats['downtrend_countermoves']['PriceChange']['mean'])
                    if avg_price_change > 0:
                        stop_price = entry_price - avg_price_change
                    else:
                        stop_price = entry_price * 0.995  # fallback to 0.5% stop
                else:
                    stop_price = entry_price * 0.995  # fallback to 0.5% stop
                    
            else:  # DownTrend
                # For bearish trades, use uptrend countermoves stats
                if 'uptrend_countermoves' in stats and 'PriceChange' in stats['uptrend_countermoves']:
                    avg_price_change = abs(stats['uptrend_countermoves']['PriceChange']['mean'])
                    if avg_price_change > 0:
                        stop_price = entry_price + avg_price_change
                    else:
                        stop_price = entry_price * 1.005  # fallback to 0.5% stop
                else:
                    stop_price = entry_price * 1.005  # fallback to 0.5% stop

            return stop_price
        
        except Exception as e:
            print(f"Error in set_stop_loss: {str(e)}")
            print(f"Stats structure: {stats}")
            # Fallback to basic 0.5% stop
            return entry_price * (0.995 if trend_type == "UpTrend" else 1.005)

    def open_position(self, candle, stats):
        """
        Open position with position sizing and correct timestamp
        """
        self.position_open = True
        self.entry_price = candle["Close"]
        self.entry_time = pd.to_datetime(candle["Datetime"])  # Store actual entry time
        self.stop_loss = self.set_stop_loss(self.entry_price, self.current_trend, stats)
        
        # Calculate position size
        self.shares = self.calculate_position_size(self.entry_price, self.stop_loss)
        
        if self.shares == 0:
            print("Warning: Zero shares calculated, skipping trade")
            self.position_open = False
            return False
        
        print(f"\nOpening Position at {self.entry_time}:")
        print(f"Entry Price: ${self.entry_price:.4f}")
        print(f"Stop Loss: ${self.stop_loss:.4f}")
        
        return True

    def calculate_position_size(self, entry_price, stop_loss):
        """
        Calculate the number of shares to buy/sell based on risk parameters
        """
        try:
            # Calculate dollar risk (how much we're willing to lose on this trade)
            dollar_risk = self.account_size * self.risk_fraction
            
            # Calculate per-share risk (difference between entry and stop)
            per_share_risk = abs(entry_price - stop_loss)
            
            # Calculate number of shares
            if per_share_risk > 0:
                shares = int(dollar_risk / per_share_risk)
            else:
                shares = 0
                print("Warning: Zero per-share risk detected")
            
            # Calculate total position value
            position_value = shares * entry_price
            
            print(f"\nPosition Size Calculation:")
            print(f"Account Size: ${self.account_size:,.2f}")
            print(f"Risk per Trade: ${dollar_risk:,.2f}")
            print(f"Entry Price: ${entry_price:.4f}")
            print(f"Stop Loss: ${stop_loss:.4f}")
            print(f"Per-Share Risk: ${per_share_risk:.4f}")
            print(f"Shares: {shares:,}")
            print(f"Total Position Value: ${position_value:,.2f}")
            
            return shares
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_current_move(self, candle, trend_direction):
        """
        Calculate the size of the current price move
        """
        if trend_direction == "UP":
            return candle['Close'] - self.trend_start_price
        else:  # DOWN
            return self.trend_start_price - candle['Close']
            
    def calculate_reversal_score(self, candle, stats, trend_direction):
        """
        Calculate reversal score based on multiple factors with detailed logging
        Returns: score (0-100), list of reasons
        """
        score = 0
        reasons = []
        detailed_log = []  # For storing detailed scoring information
        
        # Which stats to use based on trend
        stats_to_use = stats['uptrend_countermoves'] if trend_direction == "DOWN" else stats['downtrend_countermoves']
        detailed_log.append(f"Analyzing potential {trend_direction} trend reversal")
        
        # 1. Volume Analysis (0-30 points)
        avg_volume = stats_to_use['Volume']['mean']
        vol_std = stats_to_use['Volume']['std']
        
        detailed_log.append(f"\nVolume Analysis:")
        detailed_log.append(f"Current Volume: {candle['Volume']:.2f}")
        detailed_log.append(f"Average Volume: {avg_volume:.2f}")
        detailed_log.append(f"Volume Std Dev: {vol_std:.2f}")
        
        if candle["Volume"] > avg_volume:
            score += 15
            reasons.append("Volume above average")
            detailed_log.append("✓ +15 points: Volume above average")
        if candle["Volume"] > avg_volume + vol_std:
            score += 15
            reasons.append("Volume significantly above average (> 1 std)")
            detailed_log.append("✓ +15 points: Volume significantly above average")
            
        # 2. Price Movement (0-40 points)
        avg_price_change = stats_to_use['PriceChange']['mean']
        price_std = stats_to_use['PriceChange']['std']
        current_move = self.calculate_current_move(candle, trend_direction)
        
        detailed_log.append(f"\nPrice Movement Analysis:")
        detailed_log.append(f"Current Move: {current_move:.4f}")
        detailed_log.append(f"Average Move: {avg_price_change:.4f}")
        detailed_log.append(f"Move Std Dev: {price_std:.4f}")
        
        if abs(current_move) > abs(avg_price_change):
            score += 20
            reasons.append("Price movement exceeds average")
            detailed_log.append("✓ +20 points: Price movement exceeds average")
        if abs(current_move) > abs(avg_price_change + price_std):
            score += 20
            reasons.append("Price movement significantly large")
            detailed_log.append("✓ +20 points: Price movement significantly large")
            
        # 3. Duration Analysis (0-30 points)
        avg_duration = stats_to_use['Duration']['mean']
        max_duration = stats_to_use['Duration']['max']
        
        detailed_log.append(f"\nDuration Analysis:")
        detailed_log.append(f"Current Duration: {self.current_move_duration}")
        detailed_log.append(f"Average Duration: {avg_duration:.1f}")
        detailed_log.append(f"Max Duration: {max_duration}")
        
        if self.current_move_duration > avg_duration:
            score += 15
            reasons.append("Move duration above average")
            detailed_log.append("✓ +15 points: Duration above average")
        if self.current_move_duration > max_duration:
            score += 15
            reasons.append("Move duration exceeds typical max")
            detailed_log.append("✓ +15 points: Duration exceeds max")
        
        detailed_log.append(f"\nFinal Score: {score}/100")
        detailed_log.append(f"Entry threshold: {self.min_score_for_entry}")
        detailed_log.append(f"Decision: {'ENTRY' if score >= self.min_score_for_entry else 'NO ENTRY'}")
        
        # Print detailed log if score is close to or exceeds threshold
        if score >= (self.min_score_for_entry - 20):  # Show details when within 20 points of threshold
            print("\n" + "\n".join(detailed_log))
        
        return score, reasons

    def check_for_reversal(self, candle, stats):
        """
        Check if current candle indicates a reversal with enhanced logging
        Returns: (is_reversal, direction, score, reasons)
        """
        if self.last_trend_direction is None:
            self.last_trend_direction = "UP" if candle['Close'] > candle['Open'] else "DOWN"
            self.trend_start_price = candle['Close']
            self.trend_start_time = pd.to_datetime(candle['Datetime'])
            print(f"\n[{candle['Datetime']}] Initializing trend direction: {self.last_trend_direction}")
            return False, None, 0, []
            
        current_direction = "UP" if candle['Close'] > self.trend_start_price else "DOWN"
        
        if current_direction != self.last_trend_direction:
            print(f"\n[{candle['Datetime']}] Potential reversal detected:")
            print(f"Previous trend: {self.last_trend_direction}")
            print(f"Current direction: {current_direction}")
            
            score, reasons = self.calculate_reversal_score(candle, stats, current_direction)
            
            if score >= self.min_score_for_entry:
                return True, current_direction, score, reasons
                
        self.current_move_duration += 1
        return False, None, 0, []

    def manage_position(self, candle, stats):
        """
        Manage open positions with correct timestamps
        """
        if not self.position_open:
            return False, 0

        closed = False
        pnl = 0
        exit_reason = None
        
        # Calculate current P&L with proper position size
        if self.current_trend == "UpTrend":
            current_pnl = (candle['Close'] - self.entry_price) * self.shares
            if candle['Low'] <= self.stop_loss:
                closed = True
                pnl = (self.stop_loss - self.entry_price) * self.shares
                exit_reason = "Stop Loss"
        else:  # DownTrend
            current_pnl = (self.entry_price - candle['Close']) * self.shares
            if candle['High'] >= self.stop_loss:
                closed = True
                pnl = (self.entry_price - self.stop_loss) * self.shares
                exit_reason = "Stop Loss"

        # Check for trend reversal exit
        if not closed:
            reversal_score, reasons = self.calculate_reversal_score(candle, stats, 
                "DOWN" if self.current_trend == "UpTrend" else "UP")
            if reversal_score >= self.min_score_for_entry:
                closed = True
                pnl = current_pnl
                exit_reason = "Trend Reversal"

        if closed:
            # Log trade details with correct timestamps
            trade = {
                'entry_time': self.entry_time,  # Use stored entry time
                'exit_time': pd.to_datetime(candle['Datetime']),
                'entry_price': self.entry_price,
                'exit_price': candle['Close'],
                'trend_type': self.current_trend,
                'stop_loss': self.stop_loss,
                'shares': self.shares,
                'PnL': pnl,
                'exit_reason': exit_reason
            }
            self.trade_history.append(trade)
            
            # Reset position tracking
            self.position_open = False
            self.current_trend = None
            self.stop_loss = None
            self.entry_price = None
            
            # Log exit details
            print(f"\n[{candle['Datetime']}] Position Closed:")
            print(f"Exit Reason: {exit_reason}")
            print(f"P&L: ${pnl:.2f}")
            print(f"Trade Duration: {trade['exit_time'] - trade['entry_time']}")

        return closed, pnl

    def run_backtest(self, data_file):
        """
        Run backtest with enhanced position management
        """
        try:
            df = pd.read_csv(data_file)
            print(f"Processing {len(df)} candles...")
            
            trades = []
            equity_curve = [self.account_size]
            current_equity = self.account_size
            
            for i in range(len(df)):
                candle = df.iloc[i].to_dict()
                stats = self._get_stats_for_day(candle['Datetime'])
                
                # Manage existing position if open
                if self.position_open:
                    closed, pnl = self.manage_position(candle, stats)
                    if closed:
                        current_equity += pnl
                        equity_curve.append(current_equity)
                        trades.append(self.trade_history[-1])
                
                # Check for new entry if no position open
                elif not self.position_open:
                    is_reversal, direction, score, reasons = self.check_for_reversal(candle, stats)
                    if is_reversal:
                        print(f"\n[{candle['Datetime']}] Opening Position:")
                        print(f"Direction: {direction}")
                        print(f"Score: {score}")
                        print(f"Reasons: {reasons}")
                        self.current_trend = direction
                        self.open_position(candle, stats)
                
                # Update tracking variables
                if not self.position_open:
                    self.current_move_duration += 1

            results = {
                'trades': trades,
                'equity_curve': equity_curve,
                'final_equity': current_equity,
                'total_trades': len(trades),
                'profitable_trades': len([t for t in trades if t['PnL'] > 0]),
                'total_profit': sum(t['PnL'] for t in trades)
            }
            
            # Generate trade log
            self.generate_trade_log(data_file, trades, equity_curve)
            
            return results
            
        except Exception as e:
            print(f"Error in run_backtest: {str(e)}")
            return {
                'trades': [],
                'equity_curve': [self.account_size],
                'final_equity': self.account_size,
                'total_trades': 0,
                'profitable_trades': 0,
                'total_profit': 0
            }

    def generate_trade_log(self, data_file, trades, equity_curve):
        """
        Generate a detailed trade log file
        """
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(data_file), 'trade_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate log filename based on data file
            base_name = os.path.basename(data_file)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'trade_log_{base_name}_{timestamp}.txt')
            
            with open(log_file, 'w') as f:
                # Write header
                f.write("=== DETAILED TRADE LOG ===\n")
                f.write(f"Data File: {base_name}\n")
                f.write(f"Analysis Date: {timestamp}\n")
                f.write(f"Initial Account Size: ${self.account_size:,.2f}\n")
                f.write("=" * 50 + "\n\n")
                
                # Write summary statistics
                f.write("TRADING SUMMARY\n")
                f.write("-" * 20 + "\n")
                total_trades = len(trades)
                profitable_trades = len([t for t in trades if t['PnL'] > 0])
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                total_profit = sum(t['PnL'] for t in trades)
                final_equity = equity_curve[-1] if equity_curve else self.account_size
                
                f.write(f"Total Trades: {total_trades}\n")
                f.write(f"Profitable Trades: {profitable_trades}\n")
                f.write(f"Win Rate: {win_rate:.2f}%\n")
                f.write(f"Total Profit: ${total_profit:,.2f}\n")
                f.write(f"Final Equity: ${final_equity:,.2f}\n")
                f.write(f"Return: {((final_equity/self.account_size - 1) * 100):.2f}%\n\n")
                
                # Write detailed trade history
                f.write("DETAILED TRADE HISTORY\n")
                f.write("-" * 20 + "\n")
                
                for i, trade in enumerate(trades, 1):
                    f.write(f"\nTrade #{i}\n")
                    f.write(f"Entry Time: {trade['entry_time']}\n")
                    f.write(f"Exit Time: {trade['exit_time']}\n")
                    f.write(f"Duration: {trade['exit_time'] - trade['entry_time']}\n")
                    f.write(f"Direction: {trade['trend_type']}\n")
                    f.write(f"Entry Price: ${trade['entry_price']:.4f}\n")
                    f.write(f"Exit Price: ${trade['exit_price']:.4f}\n")
                    f.write(f"Stop Loss: ${trade['stop_loss']:.4f}\n")
                    f.write(f"P&L: ${trade['PnL']:.2f}\n")
                    f.write(f"Exit Reason: {trade['exit_reason']}\n")
                    f.write("-" * 40 + "\n")
                
                # Write equity curve
                f.write("\nEQUITY CURVE\n")
                f.write("-" * 20 + "\n")
                for i, equity in enumerate(equity_curve):
                    f.write(f"Point {i}: ${equity:,.2f}\n")
                
            print(f"\nTrade log generated: {log_file}")
            return log_file
            
        except Exception as e:
            print(f"Error generating trade log: {str(e)}")
            return None

# Main execution
if __name__ == "__main__":
    print("Starting backtesting script...")
    
    data_folder = r"D:\NNE_strategy\nne_strategy\data\raw\AGEN"
    print(f"Looking for data in: {data_folder}")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        exit()
        
    # Get all CSV files in the folder
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"Found {len(data_files)} CSV files:")
    for file in data_files:
        print(f"- {file}")
    
    if not data_files:
        print("No CSV files found in the data folder!")
        exit()
        
    try:
        backtest = BacktestingSystem(data_folder)
        print("Successfully initialized BacktestingSystem")
    except Exception as e:
        print(f"Error initializing BacktestingSystem: {str(e)}")
        exit()
    
    all_results = []
    for file in data_files:
        print(f"\nProcessing {file}...")
        try:
            result = backtest.run_backtest(os.path.join(data_folder, file))
            all_results.append(result)
            
            print(f"Results for {file}:")
            print(f"Total trades: {result['total_trades']}")
            print(f"Profitable trades: {result['profitable_trades']}")
            if result['total_trades'] > 0:
                win_rate = result['profitable_trades']/result['total_trades']*100
            else:
                win_rate = 0
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Total profit: ${result['total_profit']:.2f}")
            print(f"Final equity: ${result['final_equity']:.2f}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
