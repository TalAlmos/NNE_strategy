class BacktestingSystem:
    def __init__(self, data_folder, initial_account_size=100000.0, risk_fraction=0.01):
        """
        Initialize the backtesting system with the new JSON-based statistics
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

        # Load countermove analysis JSON instead of statistics.csv
        stats_file_path = r"D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis\countermove_analysis.json"
        with open(stats_file_path, 'r') as f:
            self.statistics = json.load(f)

        # We'll maintain a "live_data" DF as we receive candles
        self.live_data = None

    def _get_stats_for_day(self, day_date):
        """
        Instead of looking up daily stats, we'll now use the general patterns
        from our JSON analysis
        """
        return {
            'price_based': self.statistics['price_based'],
            'duration_based': self.statistics['duration_based']
        }

    def set_stop_loss(self, entry_price: float, trend_type: str, stats: dict):
        """
        Set stop-loss using the new JSON statistics structure
        """
        if trend_type == "UpTrend":
            # For bullish trades, use negative Large moves as guidance
            large_neg_moves = next(
                (item for item in stats['price_based']['negative'] if item['SizeGroup'] == 'Large'),
                None
            )
            if large_neg_moves:
                # Use average price movement for stop distance
                stop_distance = abs(large_neg_moves['AvgPriceAction'])
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price * 0.995  # fallback to 0.5% stop
        else:  # DownTrend
            # For bearish trades, use positive Large moves as guidance
            large_pos_moves = next(
                (item for item in stats['price_based']['positive'] if item['SizeGroup'] == 'Large'),
                None
            )
            if large_pos_moves:
                stop_distance = abs(large_pos_moves['AvgPriceAction'])
                stop_price = entry_price + stop_distance
            else:
                stop_price = entry_price * 1.005  # fallback to 0.5% stop

        return stop_price

    def open_position(self, candle, stats):
        """
        Open position with entry logic based on the new JSON statistics
        """
        self.position_open = True
        self.entry_price = candle["Close"]
        self.stop_loss = self.set_stop_loss(self.entry_price, self.current_trend, stats)
        
        # Analyze entry conditions using new statistics
        entry_reason = []
        if self.current_trend == "UpTrend":
            large_pos_moves = next(
                (item for item in stats['price_based']['positive'] if item['SizeGroup'] == 'Large'),
                None
            )
            if large_pos_moves:
                if candle["Volume"] > large_pos_moves['AvgVolume']:
                    entry_reason.append("Volume above average for large positive moves")
                if abs(self.entry_price - candle["Low"]) < large_pos_moves['AvgPriceAction']:
                    entry_reason.append("Price movement within typical large positive move range")
        else:  # DownTrend
            large_neg_moves = next(
                (item for item in stats['price_based']['negative'] if item['SizeGroup'] == 'Large'),
                None
            )
            if large_neg_moves:
                if candle["Volume"] > large_neg_moves['AvgVolume']:
                    entry_reason.append("Volume above average for large negative moves")
                if abs(self.entry_price - candle["High"]) < abs(large_neg_moves['AvgPriceAction']):
                    entry_reason.append("Price movement within typical large negative move range")
        
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
            "Stats_Used": stats
        }
        self.trade_history.append(trade)
