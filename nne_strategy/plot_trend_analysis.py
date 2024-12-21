import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Load the data from the CSV file
df = pd.read_csv('nne_strategy/data/raw/NNE_data_20241205.csv')

# Convert the 'Datetime' column to datetime objects
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract the date from the first entry in the 'Datetime' column
data_date = df['Datetime'].dt.date.iloc[0].strftime('%Y%m%d')

# Find local minima and maxima
n = 2  # Number of points to consider for local extrema
df['min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']
df['max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']

# Combine minima and maxima into a single DataFrame
extrema = pd.concat([df[['Datetime', 'min']].dropna(), df[['Datetime', 'max']].dropna()])
extrema.sort_index(inplace=True)

# Filter out extrema within the first 5 minutes
start_time = df['Datetime'].iloc[0] + pd.Timedelta(minutes=5)
extrema = extrema[extrema['Datetime'] > start_time]

# Group by hour and select the highest maxima and lowest minima
extrema['Hour'] = extrema['Datetime'].dt.floor('h')
hourly_maxima = extrema.groupby('Hour')['max'].idxmax().dropna()
hourly_minima = extrema.groupby('Hour')['min'].idxmin().dropna()

# Combine selected points and add two more
selected_points = extrema.loc[pd.concat([hourly_maxima, hourly_minima])].sort_index()

# Add two more points by selecting additional extrema
additional_points = extrema.iloc[[len(extrema)//3, 2*len(extrema)//3]]
selected_points = pd.concat([selected_points, additional_points]).sort_index()

# Add starting and ending points of the day
start_point = df.iloc[[0]][['Datetime', 'Close']].rename(columns={'Close': 'min'})
end_point = df.iloc[[-1]][['Datetime', 'Close']].rename(columns={'Close': 'max'})
selected_points = pd.concat([start_point, selected_points, end_point]).sort_index()

# Ensure distance between points is greater than 5 minutes
selected_points = selected_points.loc[selected_points['Datetime'].diff().fillna(pd.Timedelta(minutes=6)) > pd.Timedelta(minutes=5)]

# Prepare trend summary
trends = []
for i in range(len(selected_points) - 1):
    start = selected_points.iloc[i]
    end = selected_points.iloc[i + 1]
    if pd.notna(start['min']) and pd.notna(end['max']):
        trend_type = 'UpTrend'
        start_price = start['min']
        end_price = end['max']
    elif pd.notna(start['max']) and pd.notna(end['min']):
        trend_type = 'DownTrend'
        start_price = start['max']
        end_price = end['min']
    else:
        continue

    trends.append({
        'Trend Type': trend_type,
        'Start Price': start_price,
        'Start Time': start['Datetime'],
        'End Price': end_price,
        'End Time': end['Datetime']
    })

# Add trend indications to the DataFrame
df['Trend'] = None
for trend in trends:
    mask = (df['Datetime'] >= trend['Start Time']) & (df['Datetime'] <= trend['End Time'])
    df.loc[mask, 'Trend'] = trend['Trend Type']

# Save the new DataFrame with trend indications
output_file = f'nne_strategy/data/stock_trend_complete/trend_analysis_NNE_{data_date}.csv'
df.to_csv(output_file, index=False)

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(df['Datetime'], df['Close'], label='Close Price', color='blue')
plt.scatter(selected_points['Datetime'], selected_points['min'], label='Local Minima', color='red')
plt.scatter(selected_points['Datetime'], selected_points['max'], label='Local Maxima', color='green')

# Draw trend lines with alternating colors
for i in range(len(selected_points) - 1):
    if (selected_points['min'].iloc[i] and selected_points['max'].iloc[i+1]) or \
       (selected_points['max'].iloc[i] and selected_points['min'].iloc[i+1]):
        plt.plot(selected_points['Datetime'].iloc[i:i+2], selected_points[['min', 'max']].iloc[i:i+2].mean(axis=1), color='orange', linewidth=2)

plt.title('Stock Price with Trend Lines')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()