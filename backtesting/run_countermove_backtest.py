"""
Enhanced backtesting script for the NNE counter-move strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import json
import logging
from typing import List, Dict, Optional

from nne_strategy.backtesting.backtest_engine import BacktestEngine
from nne_strategy.backtesting.position_tracker import PositionTracker
from nne_strategy.countermove_strategy import CountermoveStrategy
from nne_strategy.counter_move_stats import CounterMoveStats
from nne_strategy.trend_analysis import TrendAnalysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CountermoveBacktest:
    """Backtesting engine for countermove strategy"""
    
    def __init__(self, 
                 capital: float = 100000.0,
                 entry_style: str = 'conservative',
                 risk_per_trade: float = 0.01):
        """Initialize backtest parameters"""
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        
        # Initialize components
        self.strategy = CountermoveStrategy(entry_style=entry_style)
        self.stats = CounterMoveStats()
        self.trend_analyzer = TrendAnalysis()
        self.engine = BacktestEngine(initial_capital=capital)
        
        # Performance tracking
        self.positions = []
        self.trades = []
        self.daily_pnl = {}
        
    def run_backtest(self, dates: List[str]) -> Dict:
        """Run backtest over specified dates"""
        results = []
        
        for date in dates:
            try:
                # Load and prepare data
                price_data = self._load_price_data(date)
                if price_data is None:
                    logger.warning(f"No data found for {date}")
                    continue
                
                # Run backtest for date
                daily_result = self._run_daily_backtest(date, price_data)
                if daily_result:
                    results.append(daily_result)
                    
            except Exception as e:
                logger.error(f"Error processing {date}: {str(e)}", exc_info=True)
                
        return self._aggregate_results(results)
        
    def _run_daily_backtest(self, date: str, price_data: pd.DataFrame) -> Optional[Dict]:
        """Run backtest for a single day"""
        try:
            # Analyze trends
            trends = self.trend_analyzer.identify_trends(price_data)
            
            # Process each bar
            for idx, row in price_data.iterrows():
                self._process_bar(row, trends.loc[idx])
                
            # Process end of day
            self._process_eod(date)
            
            return {
                'date': date,
                'trades': self.trades.copy(),
                'pnl': self.daily_pnl[date],
                'metrics': self._calculate_daily_metrics(date)
            }
            
        except Exception as e:
            logger.error(f"Error in daily backtest for {date}: {str(e)}")
            return None
            
    def _process_bar(self, row: pd.Series, trend: pd.Series):
        """Process single price bar"""
        # Check for entries
        if not self.positions:
            signal = self._check_entry(row, trend)
            if signal['valid']:
                self._enter_position(signal, row)
                
        # Manage existing positions
        for position in self.positions:
            self._manage_position(position, row)
            
    def _check_entry(self, row: pd.Series, trend: pd.Series) -> Dict:
        """Check for entry signals"""
        # Basic validation
        if trend['Trend'] == 'None' or trend['Strength'] < 0.5:
            return {'valid': False}
            
        # Get analysis
        analysis = self.strategy.analyze_trend(
            {'direction': trend['Trend']},
            self._get_lookback_window(row)
        )
        
        if not analysis['suitable']:
            return {'valid': False}
            
        return {
            'valid': True,
            'price': row['Close'],
            'size': self._calculate_position_size(row['Close']),
            'trend': trend['Trend'],
            'strength': trend['Strength']
        }
        
    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size based on risk"""
        return self.strategy.calculate_position_size(
            self.capital,
            price,
            self.risk_per_trade
        )
        
    @staticmethod
    def _load_price_data(date: str) -> Optional[pd.DataFrame]:
        """Load price data for date"""
        try:
            path = Path(f"data/stock_raw_data/NNE_data_{date}.csv")
            if not path.exists():
                return None
                
            data = pd.read_csv(path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            return data.set_index('Datetime')
            
        except Exception as e:
            logger.error(f"Error loading data for {date}: {str(e)}")
            return None
            
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate backtest results"""
        if not results:
            return {}
            
        total_trades = sum(len(r['trades']) for r in results)
        total_pnl = sum(r['pnl'] for r in results)
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'win_rate': self._calculate_win_rate(results),
            'daily_results': results
        }
        
    def _calculate_win_rate(self, results: List[Dict]) -> float:
        """Calculate overall win rate"""
        winning_trades = sum(
            len([t for t in r['trades'] if t['pnl'] > 0])
            for r in results
        )
        total_trades = sum(len(r['trades']) for r in results)
        
        return (winning_trades / total_trades * 100) if total_trades > 0 else 0