import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_trend_analysis(csv_file, output_image):
    # Load the data
    df = pd.read_csv(csv_file, parse_dates=['Datetime'])
    
    # Set the Datetime as the index
    df.set_index('Datetime', inplace=True)
    
    # Plot the closing price
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1)
    
    # Highlight trends
    up_trend = df[df['Trend'] == 'UpTrend']
    down_trend = df[df['Trend'] == 'DownTrend']
    
    plt.scatter(up_trend.index, up_trend['Close'], color='green', label='UpTrend', marker='^', alpha=0.6)
    plt.scatter(down_trend.index, down_trend['Close'], color='red', label='DownTrend', marker='v', alpha=0.6)
    
    # Plot the 5-minute simple moving average
    plt.plot(df.index, df['SMA5'], label='SMA5', color='orange', linestyle='--', linewidth=1)
    
    # Formatting the plot
    plt.title('Trend Analysis for NNE on 2024-12-05')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Improve date formatting on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Save the plot as a PNG image
    plt.savefig(output_image)
    plt.close()
    print(f"Plot saved as {output_image}")

# Example usage
csv_file = 'nne_strategy/data/stock_trend_complete/trend_analysis_NNE_20241205.csv'
output_image = 'trend_analysis_NNE_20241205.png'
plot_trend_analysis(csv_file, output_image)
