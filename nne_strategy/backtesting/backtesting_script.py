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
#     The Backtesting Logic with Stats Integration            #
################################################################

class BacktestingSystem:
    def __init__(self, data_folder, initial_account_size=100000.0, risk_fraction=0.01):
        """
        :param data_folder: Path to the folder containing raw CSV files.
        :param initial_account_size: Starting capital.
        :param risk_fraction: Fraction of account risked per trade (e.g., 1%).
        """
        self.data_folder = data_folder
        self.account_size = initial_account_size
        self.risk_fraction = risk_fraction

        # Strategy state
        self.position_open = False
        self.current_trend = None
        self.stop_loss = None
        self.entry_price = None
        self.trade_history = []

        # We'll load statistics.csv once here:
        stats_file_path = r"D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis\pattern_analysis_results\statistics.csv"
        self.statistics_df = pd.read_csv(stats_file_path)

        # A quick approach is to set the "Date" column as string and keep a "Total Average" row for fallback
        # We'll store them in a dict indexed by date (YYYYMMDD) or fallback to any row labeled "Total Average".
        self.stats_by_date = self._prepare_stats_lookup(self.statistics_df)

        # We'll maintain a "live_data" DF as we receive candles
        self.live_data = None

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
        """Modified version with explicit time checking"""
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
            
            # Debug print for timing
            print(f"Processing candle at: {current_time}")
            
            # Wait for enough data to establish initial trend
            if len(self.live_data) < 15:
                print(f"Waiting for more data. Current length: {len(self.live_data)}")
                continue
                
            # Define initial trend after waiting period
            if not initial_trend_set and current_time >= trend_definition_time:
                self.current_trend = define_initial_trend(self.live_data)
                initial_trend_set = True
                print(f"Initial trend defined at {current_time} as: {self.current_trend}")
                continue
            
            # After initial trend is defined
            if self.current_trend:
                new_trend, new_reversal = detect_reversal(self.live_data, {}, self.daily_stats)
                
                if new_reversal:
                    print(f"Reversal detected at {current_time}")
                    print(f"Old trend: {self.current_trend}")
                    print(f"New trend: {new_trend}")
                
                if self.position_open:
                    # Check stop-loss
                    last_close = candle["Close"]
                    if (self.current_trend == "UpTrend" and last_close <= self.stop_loss) or \
                       (self.current_trend == "DownTrend" and last_close >= self.stop_loss):
                        print(f"Stop loss hit at {current_time}. Price: {last_close}, Stop: {self.stop_loss}")
                        self.close_position(candle)
                        self.position_open = False
                    
                    # Check for trend reversal exit
                    elif new_reversal and new_trend != self.current_trend:
                        print(f"Closing position due to trend reversal at {current_time}")
                        self.close_position(candle)
                        self.position_open = False
                        self.current_trend = new_trend
                        
                        if day_trades_count < 6:
                            print(f"Opening new position in direction: {new_trend}")
                            self.open_position(candle, self.daily_stats)
                            day_trades_count += 1
                            
                else:  # No position open
                    if new_reversal and new_trend is not None and day_trades_count < 6:
                        print(f"Opening new position at {current_time}. Direction: {new_trend}")
                        self.current_trend = new_trend
                        self.open_position(candle, self.daily_stats)
                        day_trades_count += 1

        # End of day: close any open position
        if self.position_open:
            print(f"Closing position at end of day: {df['Datetime'].iloc[-1]}")
            self.close_position(df.iloc[-1])

    def open_position(self, candle, daily_stats):
        """ Open a position with detailed entry reasoning """
        self.position_open = True
        self.entry_price = candle["Close"]
        self.stop_loss = set_stop_loss(self.entry_price, self.current_trend, daily_stats)
        
        # Analyze entry conditions
        entry_reason = []
        if self.current_trend == "UpTrend":
            if candle["Close"] > daily_stats.get("Bullish Reversals_Mean_Close", 0):
                entry_reason.append("Price above typical bullish reversal mean")
            if candle["Close"] < daily_stats.get("Bullish Reversals_Mean_max", float('inf')):
                entry_reason.append("Price below typical bullish max")
        else:  # DownTrend
            if candle["Close"] < daily_stats.get("Bearish Reversals_Mean_Close", 0):
                entry_reason.append("Price below typical bearish reversal mean")
            if candle["Close"] > daily_stats.get("Bearish Reversals_Mean_min", float('-inf')):
                entry_reason.append("Price above typical bearish min")
        
        size = calculate_position_size(
            self.account_size,
            self.risk_fraction,
            self.entry_price,
            self.stop_loss
        )
        
        trade = {
            "Entry_DateTime": candle["Datetime"],
            "Entry_Price": self.entry_price,
            "Type": "LONG" if self.current_trend == "UpTrend" else "SHORT",
            "Entry_Reason": "; ".join(entry_reason) if entry_reason else "Basic trend reversal",
            "StopLoss": self.stop_loss,
            "Size": size,
            "Daily_Stats_Used": {k: v for k, v in daily_stats.items() if isinstance(v, (int, float))}
        }
        self.trade_history.append(trade)

    def close_position(self, candle, exit_reason=""):
        """ Close position with exit reasoning """
        exit_price = candle["Close"]
        if len(self.trade_history) > 0 and "Exit_Price" not in self.trade_history[-1]:
            last_trade = self.trade_history[-1]
            last_trade["Exit_DateTime"] = candle["Datetime"]
            last_trade["Exit_Price"] = exit_price
            
            # Determine exit reason if not provided
            if not exit_reason:
                if exit_price <= last_trade["StopLoss"] and last_trade["Type"] == "LONG":
                    exit_reason = "Stop-loss hit (Long)"
                elif exit_price >= last_trade["StopLoss"] and last_trade["Type"] == "SHORT":
                    exit_reason = "Stop-loss hit (Short)"
                else:
                    exit_reason = "Trend reversal detected"
            
            last_trade["Exit_Reason"] = exit_reason
            
            # Calculate PnL
            if last_trade["Type"] == "LONG":
                pnl_per_share = exit_price - last_trade["Entry_Price"]
            else:
                pnl_per_share = last_trade["Entry_Price"] - exit_price
            last_trade["PnL"] = pnl_per_share * last_trade["Size"]
            self.account_size += last_trade["PnL"]

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
            report += f"Trade #{i}\n"
            report += f"Entry:\n"
            report += f"  DateTime: {trade['Entry_DateTime']}\n"
            report += f"  Price: ${trade['Entry_Price']:.2f}\n"
            report += f"  Direction: {trade['Type']}\n"
            report += f"  Reason: {trade['Entry_Reason']}\n"
            
            if 'Exit_DateTime' in trade:
                report += f"Exit:\n"
                report += f"  DateTime: {trade['Exit_DateTime']}\n"
                report += f"  Price: ${trade['Exit_Price']:.2f}\n"
                report += f"  Reason: {trade['Exit_Reason']}\n"
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


def main():
    data_folder = r"D:\NNE_strategy\nne_strategy\data\raw"

    bt_system = BacktestingSystem(
        data_folder=data_folder,
        initial_account_size=100000.0,
        risk_fraction=0.01
    )
    bt_system.run_backtest()

    # Optionally, save the final trade results in a CSV or DB
    # pd.DataFrame(bt_system.trade_history).to_csv("backtest_results.csv", index=False)


if __name__ == "__main__":
    main() 