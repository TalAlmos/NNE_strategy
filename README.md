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