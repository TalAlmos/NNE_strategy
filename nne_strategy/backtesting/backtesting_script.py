"""
Backtesting Script for Our Strategy, Now Incorporating statistics.csv

This version enhances the baseline backtesting logic by reading daily
statistical patterns (from statistics.csv) and integrating them into
the reversal detection and stop-loss decisions.

Location of input files:
 - Candle data: D:\\NNE_strategy\\nne_strategy\\data\\raw
 - Statistics data: D:\\NNE_strategy\\nne_strategy\\data\\counter_riversal_analysis\\pattern_analysis_results\\statistics.csv

Conceptual Changes:
1. Load statistics.csv at initialization, store in self.statistics_df.
2. For each trading day, retrieve that day's row (or fallback to the "Total Average" row).
3. Pass the relevant stats to detect_reversal and set_stop_loss to refine entries/exits.
4. Adjust signals and stop thresholds based on typical bullish/bearish stats, standard deviation, etc.
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
import talib

################################################################
#                      Utility Functions                       #
################################################################

def detect_reversal(live_data: pd.DataFrame, historical_patterns: dict, daily_stats: dict):
    """
    Detect potential reversal using a simplified approach
    """
    if len(live_data) < 15:  # Need at least 15 candles
        return (None, False)
    
    # Calculate short-term momentum
    recent_closes = live_data['Close'].tail(5).values
    short_term_momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
    
    # Get current trend
    current_trend = define_initial_trend(live_data)
    
    # Define reversal thresholds (can be adjusted)
    REVERSAL_THRESHOLD = 0.002  # 0.2% price movement
    
    # Check for reversal
    if current_trend == "UpTrend" and short_term_momentum < -REVERSAL_THRESHOLD:
        return ("DownTrend", True)
    elif current_trend == "DownTrend" and short_term_momentum > REVERSAL_THRESHOLD:
        return ("UpTrend", True)
    
    return (current_trend, False)


def set_stop_loss(entry_price: float, trend_type: str, daily_stats: dict):
    """
    Set a stop-loss based on the position's direction and daily statistics.
    This is a simplified version; real logic might be more elaborate.
    """
    if trend_type == "UpTrend":
        # For bullish trades, a typical protective stop might be placed
        # below some measure of 'Bearish Reversals_Mean_low' or below entry.
        # We'll do a naive approach: 0.5 * Bearish Reversals_Standard Deviation_Low below entry
        sd_low = daily_stats.get("Bearish Reversals_Standard Deviation_Low", 0.5)
        stop_price = entry_price - sd_low
    else:
        # For bearish trades, place stop above some measure of 'Bullish Reversals_Mean_high'
        sd_high = daily_stats.get("Bullish Reversals_Standard Deviation_High", 0.5)
        stop_price = entry_price + sd_high

    return stop_price


def calculate_position_size(account_size: float, risk_fraction: float, entry_price: float, stop_price: float):
    """
    Calculate position size given:
     - account_size
     - fraction of account we are willing to risk
     - difference between entry and stop
    Returns integer size of shares.
    """
    risk_amount = account_size * risk_fraction
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share == 0:
        return 0
    return int(risk_amount / risk_per_share)


def define_initial_trend(df: pd.DataFrame, lookback_period: int = 15) -> str:
    """
    Define the initial trend using a simpler, more responsive approach for shorter timeframes
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLC data
    lookback_period : int
        Number of candles to look back (default: 15 minutes)
    
    Returns:
    --------
    str
        'UpTrend' or 'DownTrend'
    """
    if len(df) < lookback_period:
        return None
    
    # Get recent price action
    recent_data = df.tail(lookback_period)
    
    # Calculate simple moving averages
    sma_5 = recent_data['Close'].rolling(window=5).mean()
    sma_15 = recent_data['Close'].rolling(window=15).mean()
    
    # Get current price and moving averages
    current_price = recent_data['Close'].iloc[-1]
    current_sma_5 = sma_5.iloc[-1]
    current_sma_15 = sma_15.iloc[-1]
    
    # Determine trend based on price position relative to MAs
    if current_price > current_sma_5 and current_sma_5 > current_sma_15:
        return "UpTrend"
    elif current_price < current_sma_5 and current_sma_5 < current_sma_15:
        return "DownTrend"
    else:
        # If unclear, use recent price action
        price_change = recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]
        return "UpTrend" if price_change > 0 else "DownTrend"


################################################################
#                      Utility Classes                         #
################################################################

class CountermoveTracker:
    def __init__(self, trade_type: str, entry_price: float, reference_price: float, entry_time):
        self.trade_type = trade_type
        self.entry_price = entry_price
        self.reference_price = reference_price
        self.entry_time = entry_time
        self.countermove_start_time = None
        self.cumulative_volume = 0
        self.duration_minutes = 0
        self.is_active = False
        
    # ... (rest of CountermoveTracker implementation)


################################################################
#     The Backtesting Logic with Stats Integration            #
################################################################

class BacktestingSystem:
    def __init__(self, data_folder, initial_account_size=10000.0, risk_fraction=0.10, size_group='Large'):
        """Modified to initialize stats_by_date"""
        self.data_folder = data_folder
        self.account_size = initial_account_size
        self.risk_fraction = risk_fraction
        self.size_group = size_group
        
        # Initialize signal calculator
        self.signal_calculator = SignalCalculator()
        
        # Strategy state
        self.position_open = False
        self.current_trend = None
        self.stop_loss = None
        self.entry_price = None
        self.trade_history = []
        
        # Load countermove statistics
        stats_file_path = r"D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis\countermove_analysis.json"
        with open(stats_file_path, 'r') as f:
            stats = json.load(f)
        
        # Filter stats for specific size group
        self.countermove_stats = {
            'positive': next(g for g in stats['price_based']['positive'] if g['SizeGroup'] == size_group),
            'negative': next(g for g in stats['price_based']['negative'] if g['SizeGroup'] == size_group)
        }
        
        # Initialize stats_by_date
        stats_file = r"D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis\pattern_analysis_results\statistics.csv"
        if os.path.exists(stats_file):
            stats_df = pd.read_csv(stats_file)
            self.stats_by_date = self._prepare_stats_lookup(stats_df)
        else:
            print(f"Warning: Statistics file not found at {stats_file}")
            self.stats_by_date = {"Total Average": {}}
        
        # We'll maintain a "live_data" DF as we receive candles
        self.live_data = None
        
        print(f"\nInitialized backtest for {size_group} size group:")
        print(f"Positive thresholds: Duration={self.countermove_stats['positive']['AvgDuration']:.1f}min, "
              f"Price%={self.countermove_stats['positive']['AvgPricePct']:.2f}%, "
              f"Volume={self.countermove_stats['positive']['AvgVolume']:.0f}")
        print(f"Negative thresholds: Duration={self.countermove_stats['negative']['AvgDuration']:.1f}min, "
              f"Price%={self.countermove_stats['negative']['AvgPricePct']:.2f}%, "
              f"Volume={self.countermove_stats['negative']['AvgVolume']:.0f}\n")

    def _prepare_stats_lookup(self, df: pd.DataFrame) -> dict:
        """
        Convert the statistics.csv DataFrame into a dictionary keyed by date string
        or 'Total Average'.
        """
        stats_dict = {}
        # Some rows might have a "Date" like 20241209. We'll store them in stats_dict["20241209"] = row.to_dict()
        for idx, row in df.iterrows():
            date_key = str(row.get("Date", "None"))
            stats_dict[date_key] = row.to_dict()

        # We might rely on a fallback "Total Average" row if it exists
        return stats_dict

    def _get_stats_for_day(self, day_date):
        """
        Retrieve the dictionary of stats for a given day_date (format: 2024-12-09).
        If not found, fallback to "Total Average" row or an empty dict.
        """
        date_str_no_dash = day_date.strftime("%Y%m%d")  # e.g. 20241209
        if date_str_no_dash in self.stats_by_date:
            return self.stats_by_date[date_str_no_dash]
        elif "Total Average" in self.stats_by_date:
            return self.stats_by_date["Total Average"]
        else:
            return {}  # no stats available

    def backtest_single_file(self, csv_file):
        """Modified version with countermove monitoring"""
        print(f"\nStarting backtest on {csv_file}...")
        df = pd.read_csv(csv_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Reset state for new file
        self.live_data = None
        self.position_open = False
        day_trades_count = 0
        initial_trend_set = False
        
        # Get daily stats for this date
        current_date = df['Datetime'].iloc[0].date()
        self.daily_stats = self._get_stats_for_day(current_date)
        print(f"Loaded daily stats for: {current_date}")
        
        session_start_time = df['Datetime'].iloc[0].replace(hour=9, minute=30)
        trend_definition_time = session_start_time + timedelta(minutes=15)
        
        print(f"Session start time: {session_start_time}")
        print(f"Trend definition time: {trend_definition_time}")
        
        for _, candle in df.iterrows():
            current_time = pd.to_datetime(candle['Datetime'])
            
            # Skip pre-market
            if current_time.time() < session_start_time.time():
                continue
                
            # Initialize or update live_data
            self.append_new_candle(candle)
            
            # -----------------------------------------------------------------
            # (A) Update CountermoveTracker each candle for open positions
            # -----------------------------------------------------------------
            if self.position_open and self.countermove_tracker:
                self.countermove_tracker.is_active = True
                self.countermove_tracker.duration_minutes = (
                    (current_time - self.countermove_tracker.entry_time).total_seconds() / 60.0
                )
                self.countermove_tracker.cumulative_volume += candle["Volume"]

            if len(self.live_data) < 15:
                continue

            # Define initial trend after waiting period
            if not initial_trend_set and current_time >= trend_definition_time:
                self.current_trend = define_initial_trend(self.live_data)
                initial_trend_set = True
                continue
            
            if self.current_trend:
                new_trend, new_reversal = detect_reversal(self.live_data, {}, self.daily_stats)
                
                if self.position_open:
                    # Check for exit conditions (including take-profit)
                    should_exit, exit_reason = self.check_exit_conditions(candle)
                    if should_exit:
                        self.close_position(candle, exit_reason)
                        self.position_open = False
                        continue
                    
                    # ---------------------------------------------------------
                    # (B) Provide explicit exit reasons for stop-loss or reversal
                    # ---------------------------------------------------------
                    last_close = candle["Close"]
                    
                    # Stop-loss
                    if (self.current_trend == "UpTrend" and last_close <= self.stop_loss):
                        reason_str = f"Stop loss triggered at {last_close:.2f}"
                        self.close_position(candle, exit_reason=reason_str)
                        self.position_open = False
                    
                    # For DownTrend
                    elif (self.current_trend == "DownTrend" and last_close >= self.stop_loss):
                        reason_str = f"Stop loss triggered at {last_close:.2f}"
                        self.close_position(candle, exit_reason=reason_str)
                        self.position_open = False
                    
                    # Trend reversal exit
                    elif new_reversal and new_trend != self.current_trend:
                        reason_str = f"Trend reversal from {self.current_trend} to {new_trend}"
                        self.close_position(candle, exit_reason=reason_str)
                        self.position_open = False
                        self.current_trend = new_trend
                        
                        if day_trades_count < 6:
                            self.open_position(candle, self.daily_stats)
                            day_trades_count += 1
                else:
                    # Open a new position on a reversal or if you have a valid signal
                    if new_reversal and new_trend is not None and day_trades_count < 6:
                        self.current_trend = new_trend
                        self.open_position(candle, self.daily_stats)
                        day_trades_count += 1
        
        # End of day close-out
        if self.position_open:
            self.close_position(df.iloc[-1], exit_reason="End of day closure")
            self.position_open = False

    def check_entry_conditions(self, candle):
        """Check if entry conditions are met using signal weighting system"""
        if len(self.live_data) < self.signal_calculator.required_data_points:
            print(f"Insufficient data points: {len(self.live_data)} < {self.signal_calculator.required_data_points}")
            return False, None, ""
        
        total_score, entry_type, details = self.signal_calculator.get_total_score(self.live_data)
        
        # Get timestamp from index if it's a datetime index, otherwise use the last known time
        current_time = self.live_data.index[-1] if isinstance(self.live_data.index, pd.DatetimeIndex) else "current candle"
        
        print(f"\nDetailed Signal Analysis at {current_time}:")
        print(f"Current Price: {candle['Close']}")
        print(f"Trend Score: {details['trend'][0]}% ({details['trend'][1]})")
        print(f"Volume Score: {details['volume']}%")
        print(f"Price Action Score: {details['price'][0]}% ({details['price'][1]})")
        print(f"Reversal Score: {details['reversal'][0]}% ({details['reversal'][1]})")
        print(f"Total Score: {total_score}%")
        print(f"Entry Type: {entry_type}")
        
        if total_score >= self.signal_calculator.entry_threshold and entry_type:
            entry_reason = self._build_entry_reason(details)
            print(f"Entry conditions met! Score: {total_score}%, Reason: {entry_reason}")
            return True, entry_type, entry_reason
        
        print(f"Entry conditions not met. Required: {self.signal_calculator.entry_threshold}%")
        return False, None, ""

    def _build_entry_reason(self, signal_details):
        """Build detailed entry reason string"""
        reasons = []
        
        trend_score, trend_direction = signal_details["trend"]
        if trend_score > 0:
            reasons.append(f"Trend alignment ({trend_score:.1f}%): {trend_direction}")
        
        volume_score = signal_details["volume"]
        if volume_score > 0:
            reasons.append(f"Volume confirmation ({volume_score:.1f}%)")
        
        price_score, pattern_type = signal_details["price"]
        if price_score > 0:
            reasons.append(f"Price action ({price_score:.1f}%): {pattern_type}")
        
        reversal_score, reversal_type = signal_details["reversal"]
        if reversal_score > 0:
            reasons.append(f"Reversal signal ({reversal_score:.1f}%): {reversal_type}")
        
        return "; ".join(reasons)

    def open_position(self, candle, daily_stats):
        """Modified to use new signal weighting system"""
        # Check entry conditions
        should_enter, entry_type, entry_reason = self.check_entry_conditions(candle)
        
        if not should_enter:
            return False
        
        self.position_open = True
        self.entry_price = candle["Close"]
        self.reference_price = self.entry_price
        
        # Set stop loss
        self.stop_loss = set_stop_loss(self.entry_price, entry_type, daily_stats)
        
        # Initialize countermove tracker
        self.countermove_tracker = CountermoveTracker(
            trade_type=entry_type,
            entry_price=self.entry_price,
            reference_price=self.reference_price,
            entry_time=candle["Datetime"]
        )
        
        # Calculate position size
        size = calculate_position_size(
            self.account_size,
            self.risk_fraction,
            self.entry_price,
            self.stop_loss
        )
        
        # ---------------------------------------------------------
        # (C) Simple example: add a 2:1 reward to risk take-profit
        # ---------------------------------------------------------
        # This is entirely up to your strategy. You can customize.
        risk_per_share = abs(self.entry_price - self.stop_loss)
        if risk_per_share == 0:
            take_profit_price = self.entry_price  # fallback
        else:
            if entry_type == "LONG":
                take_profit_price = self.entry_price + 2 * risk_per_share
            else:  # SHORT
                take_profit_price = self.entry_price - 2 * risk_per_share
        
        # Record trade
        trade = {
            "Entry_DateTime": candle["Datetime"],
            "Entry_Price": self.entry_price,
            "Type": entry_type,
            "Entry_Reason": entry_reason,
            "StopLoss": self.stop_loss,
            "Size": size,
            "Signal_Score": self.signal_calculator.get_total_score(self.live_data)[0],
            "TakeProfit": take_profit_price  # Store in your trade dict
        }
        self.trade_history.append(trade)
        
        return True

    def close_position(self, candle, exit_reason=""):
        """Modified to include detailed exit reasoning and PnL calculation"""
        exit_price = candle["Close"]
        if len(self.trade_history) > 0 and "Exit_Price" not in self.trade_history[-1]:
            last_trade = self.trade_history[-1]
            last_trade["Exit_DateTime"] = candle["Datetime"]
            last_trade["Exit_Price"] = exit_price
            last_trade["Exit_Reason"] = exit_reason
            
            # Calculate PnL
            entry_price = last_trade["Entry_Price"]
            position_size = last_trade["Size"]
            
            if last_trade["Type"] == "LONG":
                pnl = (exit_price - entry_price) * position_size
            else:  # SHORT
                pnl = (entry_price - exit_price) * position_size
            
            last_trade["PnL"] = pnl
            
            # Calculate countermove metrics
            if self.countermove_tracker and self.countermove_tracker.is_active:
                last_trade["Countermove_Duration"] = self.countermove_tracker.duration_minutes
                last_trade["Countermove_Volume"] = self.countermove_tracker.cumulative_volume
                last_trade["Countermove_PricePct"] = (
                    (exit_price - self.countermove_tracker.reference_price) 
                    / self.countermove_tracker.reference_price * 100
                )
            
            # Optionally log the final trade PnL
            print(f"Closing {last_trade['Type']} at {exit_price:.2f}, reason: {exit_reason}, PnL: {pnl:.2f}")

    def generate_trade_report(self):
        """Generate a detailed trade summary report"""
        if not self.trade_history:
            return "No trades were executed during this period."
        
        report = "TRADING SUMMARY REPORT\n"
        report += "=" * 80 + "\n\n"
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('PnL', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('PnL', 0) < 0])
        
        report += f"Total Trades: {total_trades}\n"
        report += f"Winning Trades: {winning_trades}\n"
        report += f"Losing Trades: {losing_trades}\n"
        report += f"Win Rate: {(winning_trades/total_trades*100):.2f}%\n"
        report += f"Final P&L: ${sum(t.get('PnL', 0) for t in self.trade_history):,.2f}\n\n"
        
        report += "DETAILED TRADE LIST\n"
        report += "=" * 80 + "\n\n"
        
        for i, trade in enumerate(self.trade_history, 1):
            report += "Trade #{}\n".format(i)
            report += "Entry:\n"
            report += f"  DateTime: {trade['Entry_DateTime']}\n"
            report += f"  Price: ${trade['Entry_Price']:.2f}\n"
            report += f"  Direction: {trade['Type']}\n"
            report += f"  Reason: {trade['Entry_Reason']}\n"
            report += "Exit:\n"
            report += f"  DateTime: {trade['Exit_DateTime']}\n"
            report += f"  Price: ${trade['Exit_Price']:.2f}\n"
            report += f"  Reason: {trade.get('Exit_Reason', '')}\n"
            report += f"Trade P&L: ${trade['PnL']:.2f}\n"
            report += "-" * 40 + "\n"
        
        return report

    def append_new_candle(self, candle):
        """
        Maintain a 'live_data' DataFrame that we feed into
        detect_reversal, etc.
        """
        # Convert candle to DataFrame with explicit dtypes
        new_candle = pd.DataFrame({
            "Datetime": [candle["Datetime"]],
            "Open": [float(candle["Open"])],
            "High": [float(candle["High"])],
            "Low": [float(candle["Low"])],
            "Close": [float(candle["Close"])],
            "Volume": [int(candle["Volume"])]
        })
        
        if self.live_data is None:
            self.live_data = new_candle
        else:
            self.live_data = pd.concat([self.live_data, new_candle], ignore_index=True)
        
        # Optionally, keep only last N candles to prevent memory growth
        if len(self.live_data) > 100:  # arbitrary window size
            self.live_data = self.live_data.tail(100)

    def run_backtest(self):
        """
        Run backtest and generate report
        """
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        for csv_file in csv_files:
            self.backtest_single_file(csv_file)

        # Generate and save the report
        report = self.generate_trade_report()
        
        # Print to console
        print(report)
        
        # Save to file
        report_path = os.path.join(os.path.dirname(self.data_folder), "backtest_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Also save detailed trade data to CSV for further analysis
        results_df = pd.DataFrame(self.trade_history)
        csv_path = os.path.join(os.path.dirname(self.data_folder), "backtest_results.csv")
        results_df.to_csv(csv_path, index=False)

    def check_exit_conditions(self, candle):
        """Modified for single size group checking"""
        if not self.position_open or not self.countermove_tracker:
            return False, ""
            
        current_price = candle["Close"]
        move_type = 'negative' if self.countermove_tracker.trade_type == 'LONG' else 'positive'
        
        if move_type not in self.countermove_stats:
            print(f"Warning: {move_type} moves not found in stats")
            return False, ""
            
        stats = self.countermove_stats[move_type]
        price_change_pct = abs((current_price - self.reference_price) / self.reference_price * 100)
        
        # Check thresholds for this size group
        if self.countermove_tracker.duration_minutes > stats['AvgDuration']:
            return True, f"{self.size_group} duration exceeded: {self.countermove_tracker.duration_minutes:.1f} > {stats['AvgDuration']:.1f} min"
            
        if price_change_pct > abs(stats['AvgPricePct']):
            return True, f"{self.size_group} price movement exceeded: {price_change_pct:.2f}% > {abs(stats['AvgPricePct']):.2f}%"
            
        if self.countermove_tracker.cumulative_volume > stats['AvgVolume']:
            return True, f"{self.size_group} volume exceeded: {self.countermove_tracker.cumulative_volume:.0f} > {stats['AvgVolume']:.0f}"
        
        # ---------------------------------------------------------
        # (2) Check for take-profit
        # ---------------------------------------------------------
        if self.trade_history:
            last_trade = self.trade_history[-1]
            if "TakeProfit" in last_trade:
                take_profit_price = last_trade["TakeProfit"]
                # If it's a long trade and price exceeds take-profit
                if last_trade["Type"] == "LONG" and current_price >= take_profit_price:
                    return True, f"Take-profit triggered at {current_price:.2f}"
                # If it's a short trade and price goes below take-profit
                if last_trade["Type"] == "SHORT" and current_price <= take_profit_price:
                    return True, f"Take-profit triggered at {current_price:.2f}"
        
        return False, ""


def run_multiple_backtests():
    """Run separate backtests for each size group"""
    data_folder = r"D:\NNE_strategy\nne_strategy\data\raw"
    size_groups = ['Large', 'Medium', 'Small']
    results = {}
    
    for size_group in size_groups:
        print(f"\n{'='*80}")
        print(f"Starting backtest for {size_group} size group")
        print(f"{'='*80}\n")
        
        # Initialize backtest system for this size group
        bt_system = BacktestingSystem(
            data_folder=data_folder,
            initial_account_size=100000.0,
            risk_fraction=0.01,
            size_group=size_group
        )
        
        # Run backtest
        bt_system.run_backtest()
        
        # Generate and store report
        report = bt_system.generate_trade_report()
        results[size_group] = {
            'report': report,
            'final_pnl': sum(t.get('PnL', 0) for t in bt_system.trade_history),
            'trade_count': len(bt_system.trade_history),
            'win_rate': len([t for t in bt_system.trade_history if t.get('PnL', 0) > 0]) / len(bt_system.trade_history) if bt_system.trade_history else 0
        }
        
        # Print individual report
        print(f"\nResults for {size_group} size group:")
        print(f"Total Trades: {results[size_group]['trade_count']}")
        print(f"Win Rate: {results[size_group]['win_rate']*100:.2f}%")
        print(f"Final P&L: ${results[size_group]['final_pnl']:,.2f}")
        
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\nSize Group | Trades | Win Rate | Final P&L")
    print("-"*50)
    for size_group in size_groups:
        print(f"{size_group:9} | {results[size_group]['trade_count']:6d} | "
              f"{results[size_group]['win_rate']*100:7.2f}% | "
              f"${results[size_group]['final_pnl']:,.2f}")
    
    # Save detailed reports to files
    for size_group in size_groups:
        filename = f"backtest_report_{size_group.lower()}.txt"
        with open(filename, 'w') as f:
            f.write(results[size_group]['report'])
        print(f"\nDetailed report for {size_group} saved to {filename}")


class SignalCalculator:
    def __init__(self):
        self.lookback_period = 20
        self.rsi_period = 14
        self.required_data_points = 30
        self.entry_threshold = 45
        
    def calculate_trend_score(self, data: pd.DataFrame) -> tuple:
        # Fixed column name to match DataFrame
        trend_score = 30.0
        trend_direction = "UP" if data['Close'].iloc[-1] > data['Close'].iloc[-2] else "DOWN"
        return trend_score, trend_direction
        
    def calculate_volume_score(self, data: pd.DataFrame) -> float:
        # Fixed column name to match DataFrame
        avg_volume = data['Volume'].rolling(self.lookback_period).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        
        if current_volume > avg_volume * 1.1:
            return 25.0
        elif current_volume > avg_volume:
            return 15.0
        return 0.0
        
    def calculate_price_action(self, data: pd.DataFrame) -> tuple:
        # Fixed column names to match DataFrame
        price_score = 0.0
        pattern_type = "NONE"
        
        # Add your price action logic here
        recent_close = data['Close'].iloc[-1]
        recent_open = data['Open'].iloc[-1]
        
        if recent_close > recent_open:
            price_score = 15.0
            pattern_type = "BULLISH"
        elif recent_close < recent_open:
            price_score = 15.0
            pattern_type = "BEARISH"
            
        return price_score, pattern_type
        
    def calculate_reversal_pattern(self, data: pd.DataFrame) -> tuple:
        # Fixed column names to match DataFrame
        if len(data) < 3:
            return 0.0, "NONE"
            
        last_three_closes = data['Close'].tail(3)
        last_three_opens = data['Open'].tail(3)
        
        # Simple reversal detection
        if all(last_three_closes.iloc[i] < last_three_opens.iloc[i] for i in range(2)):
            if last_three_closes.iloc[-1] > last_three_opens.iloc[-1]:
                return 30.0, "BULLISH"
                
        if all(last_three_closes.iloc[i] > last_three_opens.iloc[i] for i in range(2)):
            if last_three_closes.iloc[-1] < last_three_opens.iloc[-1]:
                return 30.0, "BEARISH"
                
        return 0.0, "NONE"

    def get_total_score(self, data: pd.DataFrame) -> tuple:
        """Calculate total signal score and determine entry type"""
        trend_score, trend_direction = self.calculate_trend_score(data)
        volume_score = self.calculate_volume_score(data)
        price_score, pattern_type = self.calculate_price_action(data)
        reversal_score, reversal_type = self.calculate_reversal_pattern(data)
        
        total_score = trend_score + volume_score + price_score + reversal_score
        
        entry_type = None
        if total_score >= self.entry_threshold:
            if trend_direction == "UP" and reversal_type != "BEARISH":
                entry_type = "LONG"
            elif trend_direction == "DOWN" and reversal_type != "BULLISH":
                entry_type = "SHORT"
                
        return total_score, entry_type, {
            "trend": (trend_score, trend_direction),
            "volume": volume_score,
            "price": (price_score, pattern_type),
            "reversal": (reversal_score, reversal_type)
        }


if __name__ == "__main__":
    run_multiple_backtests() 