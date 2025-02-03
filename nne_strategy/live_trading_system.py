"""
Example live trading system integration with Interactive Brokers (IB API).
The code below extends the previous 'live_trading_system.py' logic to fetch 
real-time 1-minute candles from IB. You must have TWS or IB Gateway running and 
the 'ibapi' Python package installed.

Note: The sample focuses on illustrating data flow from IB to the trading logic. 
In real deployments, ensure grace in shutting down, error handling, timezone, etc.
"""

import queue
import time
import pandas as pd
from datetime import datetime, timedelta

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

# ----------------------------------------------------------------------
# 1. IBKR Client Definition
# ----------------------------------------------------------------------
class IBKRWrapper(EWrapper):
    """
    The EWrapper is where we receive incoming messages from IB.
    Here we handle real-time bar events and add them to a queue for
    retrieval in the main trading loop.
    """
    def __init__(self):
        super().__init__()
        self._barQ = queue.Queue()

    @iswrapper
    def realtimeBar(self, reqId, time_, open_, high, low, close, volume, wap, count):
        """
        Called by IBKR whenever a new real-time bar is ready, typically every 5 seconds
        or with user-defined granularity for RTH data. We'll accumulate 1-minute candles
        in the trading loop.
        """
        bar_data = {
            "reqId": reqId,
            "Datetime": datetime.fromtimestamp(time_),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        }
        # Store bar in queue for retrieval
        self._barQ.put(bar_data)

class IBKRClient(EClient):
    """
    EClient handles the connection logic, sending requests to IB.
    """
    def __init__(self, wrapper):
        super().__init__(wrapper)

class IBKRApp(IBKRWrapper, IBKRClient):
    """
    Combines EWrapper and EClient into a single class.
    """
    def __init__(self):
        IBKRWrapper.__init__(self)
        IBKRClient.__init__(self, wrapper=self)

    def connect_and_run(self, host="127.0.0.1", port=7497, clientId=123):
        """
        Connect to TWS or IB Gateway and start the message loop thread.
        """
        self.connect(host, port, clientId)
        self.run()  # this blocks. In practice, run() would be on a separate thread.

# ----------------------------------------------------------------------
# 2. Setup a method to fetch real-time data from IB
# ----------------------------------------------------------------------
class LiveTradingSystem:
    def __init__(self):
        self.ib = IBKRApp()
        # You may want to start .run() on a separate thread in production.
        # For a simple script, you might do self.ib.connect_and_run() in main().

        self.live_data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        # Placeholders for the high-level logic
        self.position_open = False
        self.current_trend = None
        self.stop_loss = None
        self.entry_price = None
        self.trades_taken = 0

        # For demonstration, request ID is fixed
        self.reqId = 1

    def request_data(self, symbol="SPY", secType="STK", exchange="SMART", currency="USD"):
        """
        Build a contract object and request real-time bars from IB. 
        You can set different bar size or whatToShow if you prefer.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = secType
        contract.exchange = exchange
        contract.currency = currency

        # Request real-time bars (5-second bars).
        # In IB, 1-minute bars in real time are approximated by concatenating 5-second bars
        # or by collecting historical data. This example uses realTimeBars method:
        self.ib.reqRealTimeBars(
            self.reqId,       # reqId
            contract,         # contract
            5,                # barSize (seconds)
            "TRADES",         # whatToShow: TRADES, MIDPOINT, etc.
            True,             # useRTH: data within regular trading hours
            []
        )

    def get_realtime_candle(self) -> dict:
        """
        A method called by the main trading loop every minute (or more frequently)
        to build a 1-minute candle from the 5-second bars. This sample approach 
        collects multiple 5-second bars, merges them into a single 1-minute bar.
        """
        start_time = datetime.now()
        bars_in_this_minute = []
        
        # Keep collecting 5-second bars until 1 minute has passed
        while (datetime.now() - start_time).total_seconds() < 60:
            try:
                bar = self.ib._barQ.get(timeout=2)  # wait up to 2s for next 5s bar
                bars_in_this_minute.append(bar)
            except queue.Empty:
                # No new bar arrived yet; keep waiting
                continue
        if not bars_in_this_minute:
            # If no bars were received (should not happen in a live environment),
            # return an empty or fallback candle
            fallback_candle = {
                "Datetime": datetime.now(),
                "Open": None,
                "High": None,
                "Low": None,
                "Close": None,
                "Volume": 0
            }
            return fallback_candle

        # Construct a single 1-minute candle from the collected 5-second bars
        candle_open = bars_in_this_minute[0]["Open"]
        candle_high = max(b["High"] for b in bars_in_this_minute)
        candle_low = min(b["Low"] for b in bars_in_this_minute)
        candle_close = bars_in_this_minute[-1]["Close"]
        candle_volume = sum(b["Volume"] for b in bars_in_this_minute)
        candle_time = bars_in_this_minute[-1]["Datetime"]  # time from the last bar

        return {
            "Datetime": candle_time,
            "Open": candle_open,
            "High": candle_high,
            "Low": candle_low,
            "Close": candle_close,
            "Volume": candle_volume
        }

    def detect_reversal(self, df, historical_patterns):
        """
        TODO: implement custom reversal detection logic comparing 
        'df' vs. 'historical_patterns'.
        Returns:
          - New trend direction (e.g., "UpTrend" or "DownTrend")
          - Boolean indicating if a major reversal is detected
        """
        # Placeholder logic
        # For example, if the latest candle's close is > the last 10 candles' average close => UpTrend
        # If < the last 10 candles' average close => DownTrend
        # Evaluate if it differs from the 'current_trend' to mark a reversal.
        if len(df) < 10:
            return (None, False)

        trailing_closes = df["Close"].tail(10)
        average_close = trailing_closes.mean()
        latest_close = df.iloc[-1]["Close"]

        new_trend = "UpTrend" if latest_close > average_close else "DownTrend"
        reversal_detected = (new_trend != self.current_trend and self.current_trend is not None)

        return (new_trend, reversal_detected)

    def set_stop_loss(self, entry_price, trend, historical_patterns):
        """
        TODO: Use historical pattern stats (volatility, typical pullback) 
        to set a dynamic stop loss boundary.
        """
        # Placeholder: 0.5% from entry
        buffer_percent = 0.005
        if trend == "UpTrend":
            return entry_price * (1.0 - buffer_percent)
        else:  # "DownTrend"
            return entry_price * (1.0 + buffer_percent)

    def calculate_position_size(self):
        """
        TODO: Implement position sizing, e.g., risk-based with account size,
        distance to stop, fraction-of-equity risk, etc.
        """
        # Placeholder: fixed size
        return 10

    def execute_trade(self, action, size):
        """
        TODO: Send an order to IB (e.g., Market Order).
        This requires building an Order object and calling self.ib.placeOrder(...).
        """
        print(f"Executing Trade: Action={action}, Size={size}")

    def close_position(self):
        """
        TODO: Submit an order to flatten the position.
        """
        print("Closing position...")

    def live_trading_system(self):
        """
        Main workflow: connect, request data, wait for 15 minutes, 
        detect first reversal, trade management, continuous monitoring, 
        end of day exit.
        """
        # Connect to IBKR
        # In a real app, consider threading for self.ib.run()
        self.ib.connect("127.0.0.1", 7497, clientId=123)
        # Start the message loop in the background (here for demonstration only).
        # In production, run() on a separate thread to avoid blocking.
        # from threading import Thread
        # app_thread = Thread(target=self.ib.run, daemon=True)
        # app_thread.start()
        self.request_data()

        # Market Hours (Example, 9:30 - 16:00)
        session_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        session_end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

        # Historical patterns loaded (placeholder)
        historical_patterns = {}

        # Wait for first 15 minutes
        while datetime.now() < session_start + timedelta(minutes=15):
            candle = self.get_realtime_candle()
            self.live_data.loc[len(self.live_data)] = candle
            time.sleep(1)  # minimal sleep to avoid busy loop

        # Detect first reversal
        self.current_trend, reversal_detected = self.detect_reversal(self.live_data, historical_patterns)

        if reversal_detected and self.current_trend is not None:
            self.position_open = True
            self.trades_taken += 1
            self.entry_price = self.live_data.iloc[-1]["Close"]
            self.stop_loss = self.set_stop_loss(self.entry_price, self.current_trend, historical_patterns)
            self.execute_trade("BUY" if self.current_trend == "UpTrend" else "SELL", size=self.calculate_position_size())

        # Continuous monitoring
        while datetime.now() < session_end:
            candle = self.get_realtime_candle()
            self.live_data.loc[len(self.live_data)] = candle

            # Check new trend
            new_trend, new_reversal = self.detect_reversal(self.live_data, historical_patterns)

            if self.position_open:
                last_close = candle["Close"]
                # Stop-loss check
                if self.current_trend == "UpTrend" and last_close <= self.stop_loss:
                    self.close_position()
                    self.position_open = False

                elif self.current_trend == "DownTrend" and last_close >= self.stop_loss:
                    self.close_position()
                    self.position_open = False

                if new_reversal and new_trend != self.current_trend:
                    # Major reversal => exit
                    self.close_position()
                    self.position_open = False
                    self.current_trend = new_trend

                    if self.trades_taken < 6:
                        self.position_open = True
                        self.trades_taken += 1
                        self.entry_price = self.live_data.iloc[-1]["Close"]
                        self.stop_loss = self.set_stop_loss(self.entry_price, self.current_trend, historical_patterns)
                        self.execute_trade("BUY" if self.current_trend == "UpTrend" else "SELL",
                                           size=self.calculate_position_size())
            else:
                # Not in a position, check if we should open one
                if new_reversal and self.trades_taken < 6:
                    self.current_trend = new_trend
                    self.position_open = True
                    self.trades_taken += 1
                    self.entry_price = self.live_data.iloc[-1]["Close"]
                    self.stop_loss = self.set_stop_loss(self.entry_price, self.current_trend, historical_patterns)
                    self.execute_trade("BUY" if self.current_trend == "UpTrend" else "SELL",
                                       size=self.calculate_position_size())

        # End of Day
        if self.position_open:
            self.close_position()


def main():
    system = LiveTradingSystem()
    system.live_trading_system()

if __name__ == "__main__":
    main()
