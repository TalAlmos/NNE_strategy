# NNE Trading Strategy

## Project Overview
A backtesting system for the NNE (Neural Network Enhanced) trading strategy that focuses on counter-trend movements.

## Architecture

### Core Components

1. **BacktestEngine** (`backtest_engine.py`)
   - Main backtesting engine that coordinates all components
   - Handles data loading, trend analysis, and trade execution
   - Manages capital and position tracking

2. **CountermoveStrategy** (`countermove_strategy.py`)
   - Implements the counter-trend trading logic
   - Evaluates trends and calculates entry/exit points
   - Manages risk parameters and position sizing

3. **CounterMoveStats** (`counter_move_stats.py`)
   - Handles statistical analysis of trend movements
   - Loads and processes historical trend statistics
   - Provides trend duration and size percentiles

4. **PositionTracker** (`position_tracker.py`)
   - Tracks open positions and their performance
   - Implements risk management rules
   - Handles position entry/exit logic

5. **PerformanceMetrics** (`performance_metrics.py`)
   - Calculates trading performance metrics
   - Tracks win rates, profits, and drawdowns
   - Generates performance reports

### Key Algorithms

1. **Trend Detection**
   - Uses SMA crossovers and price action
   - Confirms trends with volume analysis
   - Minimum trend duration requirements

2. **Counter-Move Detection**
   - Identifies retracements against main trend
   - Uses statistical thresholds for entry/exit
   - Implements dynamic target calculation

3. **Position Sizing**
   - Risk-based position sizing
   - Capital allocation rules
   - Maximum position limits

4. **Risk Management**
   - Trailing stops
   - Daily loss limits
   - Maximum drawdown protection

## Configuration

### Trading Parameters
- Entry Threshold: 0.3%
- Exit Threshold: 0.5%
- Stop Loss: 1%
- Minimum Profit: 0.2%
- Position Size: Max 20% of capital
- Minimum Time Between Trades: 5 minutes

### Statistical Parameters
- Trend Duration Percentiles: 15/30/45 minutes (25%/50%/75%)
- Move Size Percentiles: 0.3%/0.5%/0.8% (25%/50%/75%)
- Minimum Trend Duration: 15 minutes
- Minimum Price Change: $0.40

## Data Flow

1. Data Loading
   ```
   Raw Data -> BacktestEngine -> Trend Analysis
   ```

2. Strategy Execution
   ```
   Trend Analysis -> Counter-Move Detection -> Entry/Exit Signals
   ```

3. Position Management
   ```
   Signals -> Position Tracker -> Trade Execution
   ```

4. Performance Tracking
   ```
   Trade Results -> Performance Metrics -> Reports
   ```

## Usage

1. Running Backtests:
   ```python
   python backtesting/run_countermove_backtest.py
   ```

2. Analyzing Results:
   ```python
   # Results are saved in backtest_results/
   # Performance metrics are logged in performance.log
   ```

## Dependencies
- pandas
- numpy
- pathlib
- typing

## Project Structure
```

## Setup

1. Clone the repository
2. Create required data directories
3. Install dependencies:
   ```bash
   pip install pandas numpy
   ```

## Usage

1. Fetch data:
   ```bash
   python data_fetcher.py YYYYMMDD
   ```

2. Run trend analysis:
   ```bash
   python trend_finder.py YYYYMMDD
   ```

3. Analyze counter-moves:
   ```bash
   python countermove_analysis.py YYYYMMDD
   ```

4. Run backtest:
   ```bash
   python run_backtest.py YYYYMMDD [initial_capital]
   ```

## License

MIT License

Copyright (c) 2024 Tal Almos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.