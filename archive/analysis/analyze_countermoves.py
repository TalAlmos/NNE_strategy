from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_countermove(data, trend_start_time, trend_end_time, trend_direction):
    """Find the countermove after a trend with enhanced analysis"""
    # Get data after trend end
    post_trend_data = data[data['Datetime'] > trend_end_time].iloc[:30]  # Look at next 30 bars
    
    if post_trend_data.empty:
        return None
    
    # Calculate volume metrics
    avg_volume = data['Volume'].rolling(20).mean()
    
    if trend_direction == 'UpTrend':
        # After uptrend, look for downward countermove
        countermove_price = post_trend_data['Low'].min()
        countermove_idx = post_trend_data['Low'].idxmin()
        countermove_time = post_trend_data.loc[countermove_idx, 'Datetime']
        countermove_volume = post_trend_data.loc[countermove_idx, 'Volume']
    else:
        # After downtrend, look for upward countermove
        countermove_price = post_trend_data['High'].max()
        countermove_idx = post_trend_data['High'].idxmax()
        countermove_time = post_trend_data.loc[countermove_idx, 'Datetime']
        countermove_volume = post_trend_data.loc[countermove_idx, 'Volume']
    
    # Calculate volume ratio
    volume_ratio = countermove_volume / avg_volume[countermove_idx]
    
    return {
        'time': countermove_time,
        'price': countermove_price,
        'duration': (countermove_time - trend_end_time).total_seconds() / 60,
        'volume': countermove_volume,
        'volume_ratio': volume_ratio
    }

def analyze_day_countermoves(date, trends_file, raw_data_file):
    """Analyze countermoves for a single day with enhanced metrics"""
    # Load data
    trends = pd.read_csv(trends_file)
    raw_data = pd.read_csv(raw_data_file)
    
    # Convert datetime columns
    trends['start_time'] = pd.to_datetime(trends['start_time'])
    trends['end_time'] = pd.to_datetime(trends['end_time'])
    raw_data['Datetime'] = pd.to_datetime(raw_data['Datetime'])
    
    countermoves = []
    
    # Analyze each trend
    for _, trend in trends.iterrows():
        countermove = find_countermove(raw_data, trend['start_time'], trend['end_time'], trend['direction'])
        if countermove:
            price_change = countermove['price'] - trend['end_price']
            countermove_data = {
                'date': date,
                'trend_start_time': trend['start_time'].strftime('%H:%M'),
                'trend_end_time': trend['end_time'].strftime('%H:%M'),
                'trend_direction': trend['direction'],
                'trend_price_change': trend['price_change'],
                'trend_duration': trend['duration'],
                'countermove_time': countermove['time'].strftime('%H:%M'),
                'countermove_price_change': price_change,
                'countermove_duration': countermove['duration'],
                'countermove_ratio': abs(price_change / trend['price_change']) * 100,
                'countermove_speed': abs(price_change) / countermove['duration'],
                'countermove_volume_ratio': countermove['volume_ratio'],
                'countermove_momentum': abs(price_change) * countermove['volume_ratio']
            }
            countermoves.append(countermove_data)
    
    return countermoves

def plot_countermove_statistics(all_countermoves, output_dir):
    """Create visualizations of countermove statistics"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Countermove Ratios
    plt.subplot(2, 2, 1)
    ratios = [cm['countermove_ratio'] for cm in all_countermoves]
    plt.hist(ratios, bins=20)
    plt.title('Distribution of Countermove Ratios')
    plt.xlabel('Ratio (%)')
    plt.ylabel('Frequency')
    
    # Plot 2: Duration vs Ratio
    plt.subplot(2, 2, 2)
    durations = [cm['countermove_duration'] for cm in all_countermoves]
    plt.scatter(durations, ratios)
    plt.title('Duration vs Ratio')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Ratio (%)')
    
    # Plot 3: Volume Ratio vs Momentum
    plt.subplot(2, 2, 3)
    volume_ratios = [cm['countermove_volume_ratio'] for cm in all_countermoves]
    momentum = [cm['countermove_momentum'] for cm in all_countermoves]
    plt.scatter(volume_ratios, momentum)
    plt.title('Volume Ratio vs Momentum')
    plt.xlabel('Volume Ratio')
    plt.ylabel('Momentum')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / 'countermove_statistics.png')
    plt.close()

def analyze_all_countermoves():
    """Analyze countermoves across all days with enhanced analysis"""
    # Setup paths
    base_dir = Path(__file__).parent
    trend_dir = base_dir / "data" / "stock_trend_complete"
    raw_data_dir = base_dir / "data" / "stock_raw_data"
    output_dir = base_dir / "data" / "countermove_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all v3 trend analysis files
    trend_files = list(trend_dir.glob("trend_analysis_v3_NNE_*.csv"))
    
    print(f"\nAnalyzing countermoves for {len(trend_files)} days")
    print("=" * 50)
    
    all_countermoves = []
    
    for trend_file in sorted(trend_files):
        date = trend_file.stem.split('_')[-1]
        raw_data_file = raw_data_dir / f"NNE_data_{date}.csv"
        
        if not raw_data_file.exists():
            print(f"Missing raw data for {date}")
            continue
            
        print(f"\nAnalyzing countermoves for {date}...")
        
        try:
            countermoves = analyze_day_countermoves(date, trend_file, raw_data_file)
            all_countermoves.extend(countermoves)
            
            # Save daily results
            with open(output_dir / f"countermove_analysis_NNE_{date}.txt", 'w') as f:
                f.write(f"NNE Countermove Analysis for {date}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, cm in enumerate(countermoves, 1):
                    f.write(f"\nTrend #{i} Countermove:\n")
                    f.write(f"Trend: {cm['trend_start_time']} -> {cm['trend_end_time']} ({cm['trend_direction']})\n")
                    f.write(f"Trend Change: ${cm['trend_price_change']:.2f}\n")
                    f.write(f"Countermove at: {cm['countermove_time']}\n")
                    f.write(f"Countermove Change: ${cm['countermove_price_change']:.2f}\n")
                    f.write(f"Countermove Duration: {cm['countermove_duration']:.0f} minutes\n")
                    f.write(f"Countermove Ratio: {cm['countermove_ratio']:.1f}%\n")
                    f.write(f"Countermove Speed: ${cm['countermove_speed']:.3f}/min\n")
                    f.write(f"Volume Ratio: {cm['countermove_volume_ratio']:.2f}\n")
                    f.write(f"Momentum: {cm['countermove_momentum']:.2f}\n")
            
            print(f"Analysis complete for {date}")
            
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue
    
    # Create aggregate analysis
    if all_countermoves:
        # Save aggregate statistics
        with open(output_dir / "countermove_analysis_summary.txt", 'w') as f:
            f.write("NNE Countermove Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Days Analyzed: {len(trend_files)}\n")
            f.write(f"Total Countermoves: {len(all_countermoves)}\n\n")
            
            # Calculate aggregate statistics
            ratios = [cm['countermove_ratio'] for cm in all_countermoves]
            speeds = [cm['countermove_speed'] for cm in all_countermoves]
            volumes = [cm['countermove_volume_ratio'] for cm in all_countermoves]
            durations = [cm['countermove_duration'] for cm in all_countermoves]
            
            f.write("Countermove Ratio Statistics:\n")
            f.write(f"Average: {np.mean(ratios):.1f}%\n")
            f.write(f"Median: {np.median(ratios):.1f}%\n")
            f.write(f"Max: {np.max(ratios):.1f}%\n")
            f.write(f"Min: {np.min(ratios):.1f}%\n\n")
            
            f.write("Countermove Speed Statistics:\n")
            f.write(f"Average: ${np.mean(speeds):.3f}/min\n")
            f.write(f"Median: ${np.median(speeds):.3f}/min\n\n")
            
            f.write("Volume Ratio Statistics:\n")
            f.write(f"Average: {np.mean(volumes):.2f}\n")
            f.write(f"Median: {np.median(volumes):.2f}\n\n")
            
            f.write("Duration Statistics:\n")
            f.write(f"Average: {np.mean(durations):.0f} minutes\n")
            f.write(f"Median: {np.median(durations):.0f} minutes\n")
        
        # Create visualizations
        plot_countermove_statistics(all_countermoves, output_dir)
    
    print("\nCountermove analysis complete for all days")

if __name__ == "__main__":
    analyze_all_countermoves() 