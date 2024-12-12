# NNE Trading Strategy

A backtesting system for the NNE trading strategy, focusing on trend detection and counter-move analysis.

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

[Your chosen license]