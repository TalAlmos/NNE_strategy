import pandas as pd
from pathlib import Path

# Load data
data_path = Path("data/stock_raw_data/NNE_data_20241205.csv")
data = pd.read_csv(data_path)
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Calculate price changes
data['Price_Change'] = data['Close'].diff()
data['Pct_Change'] = data['Close'].pct_change() * 100

# Find major trend changes
print("\nMajor Price Movements (>0.15%):")
print("-" * 70)
print("Time      | Price  | Change | % Change | Volume | Cumulative")
print("-" * 70)

cumulative_change = 0
prev_trend = None
trend_start_price = data['Close'].iloc[0]

for i in range(1, len(data)):
    time = data['Datetime'].iloc[i].strftime('%H:%M')
    price = data['Close'].iloc[i]
    change = data['Price_Change'].iloc[i]
    pct_change = data['Pct_Change'].iloc[i]
    volume = data['Volume'].iloc[i]
    cumulative_change = price - trend_start_price
    
    # Print significant moves
    if abs(pct_change) > 0.15:
        print(f"{time:8} | ${price:6.2f} | {change:+6.2f} | {pct_change:+6.2f}% | {volume:6.0f} | ${cumulative_change:+6.2f}")
        
        # Check for trend change
        current_trend = 'Up' if change > 0 else 'Down'
        if prev_trend and current_trend != prev_trend:
            print(f"*** Trend Change: {prev_trend} -> {current_trend} ***")
            trend_start_price = price
        prev_trend = current_trend

# Print key levels
print("\nKey Price Levels:")
print("-" * 40)
print(f"Opening:     ${data['Close'].iloc[0]:.2f}")
print(f"High:        ${data['High'].max():.2f}")
print(f"Low:         ${data['Low'].min():.2f}")
print(f"Closing:     ${data['Close'].iloc[-1]:.2f}")
print(f"Total Move:  ${data['Close'].iloc[-1] - data['Close'].iloc[0]:+.2f}")

# Print time periods
print("\nKey Time Periods:")
print("-" * 40)
print("09:30-10:30: Initial downtrend")
print("10:30-11:30: Attempted recovery")
print("11:30-14:00: Gradual decline")
print("14:00-15:00: Consolidation")
print("15:00-16:00: Final decline")

# Filter data for 9:30-10:00
first_30min = data[
    (data['Datetime'].dt.strftime('%H:%M') >= '09:30') & 
    (data['Datetime'].dt.strftime('%H:%M') < '10:00')
]

print("\nAnalysis for first 30 minutes (9:30-10:00):")
print("-" * 40)
avg_price = first_30min['Close'].mean()
high_price = first_30min['High'].max()
low_price = first_30min['Low'].min()

print(f"Average Price: ${avg_price:.2f}")
print(f"High: ${high_price:.2f}")
print(f"Low: ${low_price:.2f}")

# Calculate distances
distance_to_high = abs(high_price - avg_price)
distance_to_low = abs(low_price - avg_price)

print(f"\nDistance Analysis:")
print(f"Distance to High: ${distance_to_high:.2f}")
print(f"Distance to Low: ${distance_to_low:.2f}")

if distance_to_high < distance_to_low:
    print("\nThe HIGH is closer to the average price")
else:
    print("\nThe LOW is closer to the average price") 