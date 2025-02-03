import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta

def descriptive_statistics(df):
    return {
        "Mean": df.mean(numeric_only=True),
        "Median": df.median(numeric_only=True),
        "Standard Deviation": df.std(numeric_only=True),
    }

def analyze_reversal_patterns(reversal_df, data, window=5):
    """
    Identify pre- and post-reversal statistics based on the reversal_df,
    a subset of 'data' that indicates where trend transitions (e.g., Bearish to Bullish) occur.
    This function fetches 'window'-sized slices before and after the reversal to study behavior.
    """
    patterns = []
    for index in reversal_df.index:
        try:
            pre = data.loc[index - window:index - 1]
            post = data.loc[index + 1:index + window]
            patterns.append({
                'Reversal Point': data.loc[index].to_dict(),
                'Pre-Reversal Stats': pre.mean(numeric_only=True).to_dict(),
                'Post-Reversal Stats': post.mean(numeric_only=True).to_dict(),
            })
        except KeyError:
            # Skip if window boundaries are out of range
            continue
    return patterns

def correlation_analysis(df):
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    return correlation_matrix

def plot_candlestick(df, title, output_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Datetime'], df['Close'], label='Close Price')
    plt.fill_between(df['Datetime'], df['Low'], df['High'], alpha=0.2, label='High-Low Range')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, f'{title.replace(" ", "_")}.png'))
    plt.close()

def calculate_periodic_averages(df, period='1H'):
    """
    Calculate periodic averages for numeric columns
    Parameters:
        df: DataFrame with 'Datetime' column and numeric data
        period: Resampling period (e.g., '1H' for hourly, '1D' for daily)
    """
    # Ensure Datetime is the index for resampling
    df = df.set_index('Datetime')
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate periodic averages
    periodic_avg = df[numeric_cols].resample(period).mean()
    
    # Reset index to make Datetime a column again
    periodic_avg = periodic_avg.reset_index()
    
    return periodic_avg

def process_file(input_file, output_folder):
    # Load data
    data = pd.read_csv(input_file, parse_dates=['Datetime'])
    
    # Filter data for each case
    countermoves = data[data['Action'] == 'Countermove']
    
    # Define reversal points based on trend transitions
    bullish_reversals = data[(data['Trend'].shift() == 'DownTrend') & (data['Trend'] == 'UpTrend')]
    bearish_reversals = data[(data['Trend'].shift() == 'UpTrend') & (data['Trend'] == 'DownTrend')]
    
    # Calculate descriptive statistics
    stats = {
        'Countermoves': descriptive_statistics(countermoves),
        'Bullish Reversals': descriptive_statistics(bullish_reversals),
        'Bearish Reversals': descriptive_statistics(bearish_reversals)
    }
    
    # Analyze patterns
    patterns = {
        'Bullish': analyze_reversal_patterns(bullish_reversals, data),
        'Bearish': analyze_reversal_patterns(bearish_reversals, data)
    }
    
    # Calculate correlations
    correlations = {
        'Countermoves': correlation_analysis(countermoves),
        'Bullish Reversals': correlation_analysis(bullish_reversals),
        'Bearish Reversals': correlation_analysis(bearish_reversals)
    }
    
    # Calculate periodic averages
    hourly_averages = calculate_periodic_averages(data, period='1H')
    daily_averages = calculate_periodic_averages(data, period='1D')
    
    # Save periodic averages
    hourly_averages.to_csv(os.path.join(output_folder, f'hourly_averages_{os.path.basename(input_file)}'), index=False)
    daily_averages.to_csv(os.path.join(output_folder, f'daily_averages_{os.path.basename(input_file)}'), index=False)
    
    # Generate plots
    plot_candlestick(countermoves, 'Countermoves', output_folder)
    plot_candlestick(bullish_reversals, 'Bullish Reversals', output_folder)
    plot_candlestick(bearish_reversals, 'Bearish Reversals', output_folder)
    
    return stats, patterns, correlations

def process_all_files(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CSV files
    all_files = glob.glob(os.path.join(input_folder, 'counter_riversal_NNE_*.csv'))
    
    # Process each file and collect results
    all_results = []
    for file in all_files:
        date = os.path.basename(file).split('_')[3].split('.')[0]
        stats, patterns, correlations = process_file(file, output_folder)
        
        result = {
            'Date': date,
            'Statistics': stats,
            'Patterns': patterns,
            'Correlations': correlations
        }
        all_results.append(result)
    
    # Save results to files
    save_results(all_results, output_folder)

def save_results(results, output_folder):
    # Save statistics
    stats_df = pd.DataFrame([{
        'Date': r['Date'],
        **{f"{k}_{stat}_{metric}": value 
           for k, stats in r['Statistics'].items()
           for stat, values in stats.items()
           for metric, value in values.items()}
    } for r in results])

    # Add total averages row
    averages = stats_df.select_dtypes(include=[np.number]).mean()
    stats_df.loc['Average'] = ['Total Average'] + list(averages)
    
    # Save to CSV
    stats_df.to_csv(os.path.join(output_folder, 'statistics.csv'), index=False)
    
    # Save correlation matrices
    for result in results:
        date = result['Date']
        for corr_type, corr_matrix in result['Correlations'].items():
            filename = f'correlation_{corr_type}_{date}.csv'
            corr_matrix.to_csv(os.path.join(output_folder, filename))

def calculate_temporary_pullback_stats(df):
    # Filter for rows where Action is Countermove
    pullbacks = df[df['Action'] == 'Countermove']

    # Calculate average time by converting to seconds since midnight
    time_of_day = pullbacks['Datetime'].dt.hour * 3600 + pullbacks['Datetime'].dt.minute * 60 + pullbacks['Datetime'].dt.second
    avg_seconds = time_of_day.mean()
    avg_time = str(timedelta(seconds=int(avg_seconds)))  # Convert to string representation

    # Calculate average price change and volume
    avg_price_change = (pullbacks['Close'] - pullbacks['Open']).mean()
    avg_volume = pullbacks['Volume'].mean()

    return avg_time, avg_price_change, avg_volume

if __name__ == "__main__":
    input_folder = r'D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis'
    output_folder = os.path.join(input_folder, 'pattern_analysis_results')
    
    process_all_files(input_folder, output_folder) 