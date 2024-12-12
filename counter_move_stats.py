import os
import glob
import pandas as pd
from datetime import datetime

class CounterMoveStats:
    def __init__(self, data_directory=None):
        """
        Initialize counter-move statistics from historical data
        
        Args:
            data_directory (str): Directory containing historical counter-move analysis files
        """
        if data_directory is None:
            # Get script directory and create correct path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_directory = os.path.join(script_dir, 'data', 'countermove_dataset')
        
        self.data_directory = data_directory
        self.stats = self._load_historical_data(data_directory)
        
    def _parse_duration(self, duration_line):
        """
        Parse duration string from the format "Duration: 0 days 00:07:00"
        
        Args:
            duration_line (str): Duration line from the file
            
        Returns:
            int: Total duration in minutes
        """
        try:
            # Remove "Duration: " prefix
            duration_str = duration_line.split("Duration: ")[1]
            
            # Split into days and time parts
            days_part, time_part = duration_str.split(" days ")
            days = int(days_part)
            
            # Parse time HH:MM:SS
            hours, minutes, _ = time_part.strip().split(":")
            
            # Calculate total minutes
            total_minutes = (days * 24 * 60) + (int(hours) * 60) + int(minutes)
            return total_minutes
            
        except Exception as e:
            print(f"Error parsing duration string '{duration_line}': {e}")
            return 0
    
    def _load_historical_data(self, directory):
        """
        Load and process all historical counter-move analysis files
        
        Args:
            directory (str): Directory path containing analysis files
            
        Returns:
            dict: Processed statistics for both trend types
        """
        # Initialize statistics structure
        all_stats = {
            'UpTrend': {
                'durations': [],
                'price_changes': [],
                'volume_changes': [],
                'success_rate': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'avg_price_change': 0,
                'max_price_change': 0,
                'total_counter_moves': 0
            },
            'DownTrend': {
                'durations': [],
                'price_changes': [],
                'volume_changes': [],
                'success_rate': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'avg_price_change': 0,
                'max_price_change': 0,
                'total_counter_moves': 0
            }
        }
        
        try:
            # Find all analysis files
            pattern = os.path.join(directory, "*.txt")
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: No historical data files found in {directory}")
                return all_stats
            
            print(f"Found {len(files)} historical data files in {directory}")
            
            # Process each file
            for file in files:
                print(f"Processing file: {os.path.basename(file)}")
                with open(file, 'r') as f:
                    lines = f.readlines()
                    current_trend = None
                    i = 0
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        if not line:
                            i += 1
                            continue
                        
                        # Identify trend and get duration
                        if "Direction:" in line:
                            current_trend = line.split(":")[1].strip()
                            
                            # Find Duration line
                            duration_line = None
                            j = i + 1
                            while j < len(lines) and "Duration:" not in lines[j]:
                                j += 1
                            if j < len(lines):
                                duration_line = lines[j].strip()
                                
                            if duration_line:
                                duration_minutes = self._parse_duration(duration_line)
                                if duration_minutes > 0:
                                    all_stats[current_trend]['durations'].append(duration_minutes)
                            
                            i = j + 1
                            continue
                        
                        # Process counter-moves (keep existing counter-move processing)
                        if "Counter-move #" in line:
                            if current_trend not in ['UpTrend', 'DownTrend']:
                                i += 1
                                continue
                            
                            # Read counter-move details
                            counter_move = {}
                            for j in range(1, 11):  # Read next 10 lines
                                if i + j >= len(lines):
                                    break
                                
                                detail_line = lines[i + j].strip()
                                
                                if "Counter-move size:" in detail_line:
                                    size = float(detail_line.split("$")[1].strip())
                                    counter_move['size'] = size
                                    all_stats[current_trend]['price_changes'].append(size)
                                elif "Percentage of trend:" in detail_line:
                                    percentage = float(detail_line.split(":")[1].strip().replace('%', ''))
                                    counter_move['percentage'] = percentage
                            
                            all_stats[current_trend]['total_counter_moves'] += 1
                            i += 11  # Skip processed lines
                        else:
                            i += 1
            
            # Calculate statistics
            for trend_type in ['UpTrend', 'DownTrend']:
                stats = all_stats[trend_type]
                if stats['price_changes']:  # Check if we have data
                    # Duration calculations
                    if stats['durations']:
                        stats['avg_duration'] = sum(stats['durations']) / len(stats['durations'])
                        stats['max_duration'] = max(stats['durations'])
                    else:
                        stats['avg_duration'] = 0
                        stats['max_duration'] = 0
                    
                    # Keep existing price change calculations
                    stats['avg_price_change'] = sum(stats['price_changes']) / len(stats['price_changes'])
                    stats['max_price_change'] = max(stats['price_changes'])
                    
                    # Success rate calculation
                    successful_moves = len([p for p in stats['price_changes'] if p > 0])
                    stats['success_rate'] = successful_moves / len(stats['price_changes'])
                    
                    # Calculate percentiles
                    if len(stats['price_changes']) >= 4:
                        stats['price_change_percentiles'] = {
                            '25': self._calculate_percentile(stats['price_changes'], 25),
                            '50': self._calculate_percentile(stats['price_changes'], 50),
                            '75': self._calculate_percentile(stats['price_changes'], 75)
                        }
                        if stats['durations']:
                            stats['duration_percentiles'] = {
                                '25': self._calculate_percentile(stats['durations'], 25),
                                '50': self._calculate_percentile(stats['durations'], 50),
                                '75': self._calculate_percentile(stats['durations'], 75)
                            }
                else:
                    print(f"Warning: No data found for {trend_type}")
        
        except Exception as e:
            print(f"Error processing historical data: {e}")
            import traceback
            traceback.print_exc()
            return all_stats
        
        return all_stats
    
    def _calculate_percentile(self, data, percentile):
        """
        Calculate percentile value from a list of numbers
        """
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile / 100
        return sorted_data[int(index)]
    
    def get_trend_stats(self, trend_type):
        """
        Get statistics for a specific trend type
        
        Args:
            trend_type (str): 'UpTrend' or 'DownTrend'
            
        Returns:
            dict: Statistics for the specified trend type
        """
        return self.stats.get(trend_type, {})
    
    def print_summary(self):
        """
        Print a summary of all counter-move statistics
        """
        print("\nCounter-Move Statistics Summary")
        print("=" * 40)
        
        for trend_type in ['UpTrend', 'DownTrend']:
            stats = self.stats[trend_type]
            print(f"\n{trend_type}:")
            print("-" * 20)
            print(f"Total Counter-Moves: {stats['total_counter_moves']}")
            print(f"Average Duration: {stats['avg_duration']:.2f} minutes")
            print(f"Max Duration: {stats['max_duration']:.2f} minutes")
            print(f"Average Price Change: ${stats['avg_price_change']:.2f}")
            print(f"Max Price Change: ${stats['max_price_change']:.2f}")
            print(f"Success Rate: {stats['success_rate']*100:.1f}%")
            
            if 'duration_percentiles' in stats:
                print("\nDuration Percentiles:")
                for p, v in stats['duration_percentiles'].items():
                    print(f"{p}th: {v:.2f} minutes")
            
            if 'price_change_percentiles' in stats:
                print("\nPrice Change Percentiles:")
                for p, v in stats['price_change_percentiles'].items():
                    print(f"{p}th: ${v:.2f}")

if __name__ == "__main__":
    # Example usage
    stats = CounterMoveStats()
    stats.print_summary() 