import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from datetime import datetime, date
import numpy as np
from scipy.signal import argrelextrema
from typing import Optional
from nne_strategy.data.data_fetcher import DataFetcher  # For "Data Processing" screen
from nne_strategy.analyze_countermoves import (
    find_countermoves,
    analyze_countermove_segments,
    read_data
)
from nne_strategy.countermoves_reversal_analysis import (
    analyze_trends,
    process_all_files
)
import tempfile  # For temporary file handling in single-file mode
import io

# Directories used for the "Trend Analysis" screen
RAW_DATA_FOLDER = r"D:\NNE_strategy\nne_strategy\data\raw"
TREND_FOLDER = r"D:\NNE_strategy\nne_strategy\data\trend"
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
os.makedirs(TREND_FOLDER, exist_ok=True)


###############################################################################
# HELPERS
###############################################################################

def parse_backtest_report(report_text: str):
    """
    Parses a backtest report text and returns a list of trades.
    Each trade is a dict with keys:
      - trade_number
      - entry_time
      - entry_price
      - entry_direction
      - entry_reason
      - exit_time
      - exit_price
      - exit_reason
      - pnl
    """
    trade_number_pattern = re.compile(r"Trade #(\d+)")
    date_time_pattern = re.compile(r"DateTime:\s+(\S+\s+\S+)")
    price_pattern = re.compile(r"Price:\s+\$(\S+)")
    direction_pattern = re.compile(r"Direction:\s+(LONG|SHORT)")
    reason_pattern = re.compile(r"Reason:\s+(.+)")
    pnl_pattern = re.compile(r"Trade P&L:\s+\$(\S+)")

    trades = []
    lines = report_text.splitlines()
    current_trade = {}
    mode = None  # "entry" or "exit"

    for line in lines:
        # Detect start of a trade
        match_trade_num = trade_number_pattern.search(line)
        if match_trade_num:
            if current_trade.get("trade_number"):
                trades.append(current_trade)
            current_trade = {}
            current_trade["trade_number"] = match_trade_num.group(1)
            mode = None
            continue

        # Detect entry or exit blocks
        if line.strip().startswith("Entry:"):
            mode = "entry"
            continue
        if line.strip().startswith("Exit:"):
            mode = "exit"
            continue

        # Parse line content by mode
        if mode == "entry":
            dt = date_time_pattern.search(line)
            if dt:
                current_trade["entry_time"] = dt.group(1)

            p = price_pattern.search(line)
            if p:
                current_trade["entry_price"] = p.group(1)

            d = direction_pattern.search(line)
            if d:
                current_trade["entry_direction"] = d.group(1)

            r = reason_pattern.search(line)
            if r:
                current_trade["entry_reason"] = r.group(1)

        elif mode == "exit":
            dt = date_time_pattern.search(line)
            if dt:
                current_trade["exit_time"] = dt.group(1)

            p = price_pattern.search(line)
            if p:
                current_trade["exit_price"] = p.group(1)

            r = reason_pattern.search(line)
            if r:
                current_trade["exit_reason"] = r.group(1)

        # Parse P&L
        match_pnl = pnl_pattern.search(line)
        if match_pnl:
            current_trade["pnl"] = match_pnl.group(1)

    # Append the last trade if it exists
    if current_trade.get("trade_number"):
        trades.append(current_trade)

    return trades


def load_minute_data(csv_file) -> pd.DataFrame:
    """
    Reads any raw CSV file into a DataFrame, ensuring Datetime is parsed and sorted.
    """
    df = pd.read_csv(csv_file)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.sort_values(by="Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


###############################################################################
# SCREEN: BACKTESTING REPORT
###############################################################################

def show_backtesting_report_screen():
    """Displays the existing Backtesting Report UI."""
    st.header("Backtesting Analysis")

    st.subheader("1) Upload Backtest Report Text File")
    report_file = st.file_uploader("Select your backtest_report.txt", type="txt")
    trades = []
    if report_file is not None:
        report_text = report_file.read().decode("utf-8")
        trades = parse_backtest_report(report_text)

        if trades:
            st.success(f"Parsed {len(trades)} trade(s).")
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df)
        else:
            st.warning("No trades found in the report.")

    st.subheader("2) (Optional) Upload Raw Minute Data CSV for Plot")
    csv_file = st.file_uploader("Select your raw data CSV (e.g., NNE_data_YYYYMMDD.csv)", type="csv")
    if csv_file is not None:
        df = load_minute_data(csv_file)
        st.write(f"Loaded {len(df)} rows of data.")
        st.dataframe(df.head(10))

        # Create simple line plot
        fig = px.line(df, 
                     x="Datetime", 
                     y="Close",
                     title="1-Minute Price Chart")
        
        fig.update_layout(
            yaxis_title="Price",
            xaxis_title="Time",
            showlegend=True,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)


###############################################################################
# SCREEN: DATA PROCESSING (Fetch new data)
###############################################################################

def show_data_processing_screen():
    """
    Displays the Data Processing UI, integrating functionality from data_fetcher.py.
    Allows user to fetch intraday data for a single stock or multiple stocks from an Excel file.
    """
    st.header("Data Fetcher")

    st.write("""
    Use this screen to fetch intraday data for stocks. You can either:
    1. Enter a single stock symbol directly
    2. Upload an Excel file containing multiple stock symbols
    """)

    # Select input mode
    input_mode = st.radio("Select Input Mode", ["Single Stock", "Multiple Stocks (Excel)"])
    
    if input_mode == "Single Stock":
        ticker = st.text_input("Stock Symbol", value="AAPL")
        tickers = [ticker] if ticker else []
    else:
        st.write("Upload an Excel file with a column named 'Ticker' containing stock symbols")
        excel_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
        
        if excel_file:
            try:
                df = pd.read_excel(excel_file)
                if 'Ticker' not in df.columns:
                    st.error("Excel file must contain a column named 'Ticker'")
                    return
                tickers = df['Ticker'].dropna().unique().tolist()
                st.success(f"Found {len(tickers)} unique stock symbols")
                st.write("Tickers found:", ", ".join(tickers))
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return
        else:
            tickers = []

    # Date selection
    single_or_range = st.radio("Select Fetch Mode", ["Single Date", "Date Range"])

    # Date Input
    if single_or_range == "Single Date":
        selected_date = st.date_input("Select Date", date.today())
        start_date_str = selected_date.strftime("%Y%m%d")
        end_date_str = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date.today())
        with col2:
            end_date = st.date_input("End Date", date.today())

        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

    # Fetch data on button click
    if st.button("Fetch Data") and tickers:
        with st.spinner("Fetching data..."):
            try:
                fetcher = DataFetcher()  # Uses default config
                total_files = 0
                failed_tickers = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, ticker in enumerate(tickers):
                    # Create stock-specific directory
                    stock_dir = os.path.join(RAW_DATA_FOLDER, ticker)
                    os.makedirs(stock_dir, exist_ok=True)

                    status_text.text(f"Processing {ticker} ({idx + 1}/{len(tickers)})")
                    
                    try:
                        results = fetcher.fetch_date_range(
                            ticker,
                            start_date_str,
                            end_date_str if single_or_range == "Date Range" else None
                        )
                        
                        if results:
                            # Save each day's data to a separate file
                            for date_str, data_df in results.items():
                                output_file = os.path.join(stock_dir, f"raw_{ticker}_{date_str}.csv")
                                data_df.to_csv(output_file, index=False)
                                total_files += 1
                        else:
                            failed_tickers.append(ticker)

                    except Exception as e:
                        failed_tickers.append(f"{ticker} (Error: {str(e)})")

                    progress_bar.progress((idx + 1) / len(tickers))

                # Final status report
                if total_files > 0:
                    st.success(f"Successfully saved {total_files} data files.")
                
                if failed_tickers:
                    st.warning("Failed to fetch data for the following tickers:")
                    for ticker in failed_tickers:
                        st.write(f"- {ticker}")

            except Exception as e:
                st.error(f"Error during data fetching: {str(e)}")


###############################################################################
# SCREEN: TREND ANALYSIS
###############################################################################

def ensure_min_max_alternating(selected_points: pd.DataFrame) -> pd.DataFrame:
    """
    Makes sure that min and max are strictly alternating. 
    If two consecutive points are both minima or both maxima, the second is dropped.
    Returns a filtered DataFrame that enforces alternation.
    """
    selected_points["type"] = selected_points.apply(
        lambda row: "min" if not pd.isna(row["min"]) else "max" if not pd.isna(row["max"]) else "none",
        axis=1
    )
    filtered_rows = []
    last_type = None
    for _, row in selected_points.iterrows():
        if row["type"] != last_type and row["type"] != "none":
            filtered_rows.append(row)
            last_type = row["type"]
    return pd.DataFrame(filtered_rows).drop(columns=["type"])


def add_trend_indications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds trend indications based on minima and maxima points.
    From minima to maxima = UpTrend
    From maxima to minima = DownTrend
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize Trend column
    df['Trend'] = None
    
    # Get points where either min or max is not null
    points = df[df['min'].notnull() | df['max'].notnull()].copy()
    points['is_min'] = points['min'].notnull()
    
    # Process each pair of consecutive points
    for i in range(len(points) - 1):
        current_point = points.iloc[i]
        next_point = points.iloc[i + 1]
        
        # Get the indices for the range in the original dataframe
        start_idx = df.index[df['Datetime'] == current_point['Datetime']][0]
        end_idx = df.index[df['Datetime'] == next_point['Datetime']][0]
        
        # Determine trend
        if current_point['is_min'] and pd.notnull(next_point['max']):
            # From min to max = UpTrend
            df.loc[start_idx:end_idx, 'Trend'] = 'UpTrend'
        elif pd.notnull(current_point['max']) and next_point['is_min']:
            # From max to min = DownTrend
            df.loc[start_idx:end_idx, 'Trend'] = 'DownTrend'
    
    return df


def show_trend_analysis_screen():
    """Displays the Trend Analysis UI with improved step-by-step workflow."""
    st.header("Trend Analysis")

    # Step 1: File Upload
    st.subheader("1️⃣ Upload Raw Stock Data")
    uploaded_file = st.file_uploader("Upload a CSV file containing stock data", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Step 2: Initial Detection Settings
        st.subheader("2️⃣ Configure Point Detection")
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Number of points to consider for local extrema", 
                              min_value=1, max_value=5, value=2)
        with col2:
            min_gap = st.number_input("Minimum minutes between points", 
                                    min_value=1, max_value=30, value=5)

        # Find local minima and maxima
        df['min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']
        df['max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']
        
        # Process first and last points
        first_point = df.iloc[0]
        last_point = df.iloc[-1]
        
        # Add first point (09:30)
        if first_point['Datetime'].time() == pd.Timestamp('09:30').time():
            next_min_idx = df[pd.notna(df['min'])].index[0] if len(df[pd.notna(df['min'])]) > 0 else None
            next_max_idx = df[pd.notna(df['max'])].index[0] if len(df[pd.notna(df['max'])]) > 0 else None
            
            if next_min_idx is not None and next_max_idx is not None:
                if next_min_idx < next_max_idx:
                    df.loc[0, 'max'] = first_point['Close']
                else:
                    df.loc[0, 'min'] = first_point['Close']
        
        # Add last point (15:59)
        if last_point['Datetime'].time() == pd.Timestamp('15:59').time():
            prev_min_idx = df[pd.notna(df['min'])].index[-1] if len(df[pd.notna(df['min'])]) > 0 else None
            prev_max_idx = df[pd.notna(df['max'])].index[-1] if len(df[pd.notna(df['max'])]) > 0 else None
            
            if prev_min_idx is not None and prev_max_idx is not None:
                if prev_min_idx > prev_max_idx:
                    df.loc[len(df)-1, 'max'] = last_point['Close']
                else:
                    df.loc[len(df)-1, 'min'] = last_point['Close']

        # Process extrema points
        extrema = pd.concat([df[['Datetime', 'min']].dropna(), df[['Datetime', 'max']].dropna()])
        extrema.sort_index(inplace=True)
        
        # Group by hour and select the highest maxima and lowest minima
        extrema['Hour'] = extrema['Datetime'].dt.floor('h')
        hourly_maxima = extrema.groupby('Hour')['max'].idxmax().dropna()
        hourly_minima = extrema.groupby('Hour')['min'].idxmin().dropna()
        
        # Combine selected points
        selected_points = extrema.loc[pd.concat([hourly_maxima, hourly_minima])].sort_index()
        
        # Add additional points at 1/3 and 2/3 of the data
        if len(extrema) >= 3:
            additional_indices = [len(extrema)//3, 2*len(extrema)//3]
            additional_points = extrema.iloc[additional_indices]
            selected_points = pd.concat([selected_points, additional_points]).sort_index()
        
        # Make sure min and max are alternating
        selected_points = ensure_min_max_alternating(selected_points)
        
        # Ensure distance between points is greater than 5 minutes
        selected_points = selected_points.loc[
            selected_points['Datetime'].diff().fillna(pd.Timedelta(minutes=6)) > pd.Timedelta(minutes=5)
        ]

        # Step 3: Show Initial Plot
        st.subheader("3️⃣ Review Auto-Detected Points")
        
        def plot_points(points_df):
            fig = go.Figure()
            
            # Base price line
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df["Close"],
                    mode="lines",
                    name="Close Price"
                )
            )

            # Plot minima and maxima
            min_points = points_df[pd.notna(points_df["min"])]
            max_points = points_df[pd.notna(points_df["max"])]
            
            if not min_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=min_points["Datetime"],
                        y=min_points["min"],
                        mode="markers",
                        marker=dict(color="red", size=6),
                        name="Local Minima"
                    )
                )
            
            if not max_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=max_points["Datetime"],
                        y=max_points["max"],
                        mode="markers",
                        marker=dict(color="green", size=6),
                        name="Local Maxima"
                    )
                )

            # Draw trend lines - improved version
            points_df = points_df.sort_values('Datetime')  # Ensure points are in chronological order
            
            for i in range(len(points_df) - 1):
                start = points_df.iloc[i]
                end = points_df.iloc[i + 1]
                
                # Get the y-values, handling both min and max columns
                start_y = start['min'] if pd.notna(start['min']) else start['max']
                end_y = end['min'] if pd.notna(end['min']) else end['max']
                
                # Only draw line if we have valid points
                if pd.notna(start_y) and pd.notna(end_y):
                    fig.add_trace(
                        go.Scatter(
                            x=[start["Datetime"], end["Datetime"]],
                            y=[start_y, end_y],
                            mode="lines",
                            line=dict(color="orange", width=2),
                            showlegend=False
                        )
                    )

            fig.update_layout(
                title="Price Trend Analysis",
                xaxis_title="Time",
                yaxis_title="Price",
                height=600,
                showlegend=True,
                hovermode='closest'
            )
            
            return fig

        # Initial plot with processed points
        st.plotly_chart(plot_points(selected_points), use_container_width=True, key="initial_plot")

        # Step 4: Point Editing
        st.subheader("4️⃣ Edit Points")
        
        # Initialize session state for edited points if not exists
        if 'edited_points' not in st.session_state:
            st.session_state.edited_points = selected_points.copy()

        # Edit points
        edited_points = st.data_editor(
            st.session_state.edited_points,
            use_container_width=True,
            num_rows="dynamic",
            key="points_editor",
            column_config={
                "Datetime": st.column_config.DatetimeColumn(
                    "Datetime",
                    format="MM/DD/YYYY HH:mm:ss",
                    step=60
                ),
                "min": "Minimum",
                "max": "Maximum"
            },
            on_change=lambda: plot_points(st.session_state.edited_points)
        )

        # Add new point button
        if st.button("Add New Point"):
            new_point = pd.DataFrame({
                'Datetime': [df['Datetime'].iloc[len(df)//2]],
                'min': [None],
                'max': [None]
            })
            st.session_state.edited_points = pd.concat([edited_points, new_point]).sort_values('Datetime')

        # Step 5: Auto-refresh plot with edited points
        st.subheader("5️⃣ Updated Visualization")
        st.plotly_chart(plot_points(edited_points), use_container_width=True, key="updated_plot")

        # Save Results
        st.subheader("📥 Save Results")
        # Get stock name and date from the uploaded file name
        if uploaded_file is not None:
            try:
                # Extract stock name from file name
                stock_name = uploaded_file.name.split('_')[1] if '_' in uploaded_file.name else 'unknown'
                
                # Get the date from the DataFrame
                file_date = df['Datetime'].iloc[0].strftime('%Y-%m-%d')
                
                # Create filename in required format
                save_name = f"trend_{stock_name}_{file_date}"
                
                if st.button("Save Analysis"):
                    try:
                        # Prepare the full save path
                        save_path = os.path.join(TREND_FOLDER, f"{save_name}.csv")
                        
                        # Save the processed DataFrame with all analysis results
                        processed_df = df.copy()
                        processed_df['min'] = None
                        processed_df['max'] = None
                        
                        # Sort points chronologically and add min/max points
                        edited_points = edited_points.sort_values('Datetime')
                        for _, row in edited_points.iterrows():
                            match_idx = processed_df[processed_df['Datetime'] == row['Datetime']].index
                            if len(match_idx) > 0:
                                if pd.notna(row['min']):
                                    processed_df.loc[match_idx[0], 'min'] = row['min']
                                if pd.notna(row['max']):
                                    processed_df.loc[match_idx[0], 'max'] = row['max']
                        
                        # Use the existing add_trend_indications function to properly set trends
                        processed_df = add_trend_indications(processed_df)
                        
                        # Save to CSV
                        processed_df.to_csv(save_path, index=False)
                        
                        # Show success message with file location
                        st.success(f"Analysis saved successfully to:\n{save_path}")
                        
                        # Optional: Add download button
                        with open(save_path, 'rb') as f:
                            st.download_button(
                                label="Download CSV",
                                data=f,
                                file_name=f"{save_name}.csv",
                                mime='text/csv'
                            )
                            
                    except Exception as e:
                        st.error(f"Error saving analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error processing file name or date: {str(e)}")
        else:
            st.warning("Please upload a file first")


###############################################################################
# ADDITIONAL IMPORTS FOR REVERSAL ANALYSIS
###############################################################################

def show_countermoves_reversal_analysis_screen():
    """
    Displays a Streamlit UI for analyzing trend data to identify Reversals and Countermoves.
    """
    st.title("Trend Analysis & Reversal Detection")
    st.markdown("Analyze trend data, identify reversals and countermoves, and visualize results dynamically.")

    # Select mode: Single File or Batch
    mode_selection = st.sidebar.radio("Processing Mode", ["Single File", "Batch Processing"])

    if mode_selection == "Single File":
        st.subheader("Single File Processing")

        # File Uploader
        uploaded_file = st.file_uploader("Upload a CSV file containing columns [Datetime, Close, Trend].", type="csv")
        date_str = st.text_input("Enter Date (YYYYMMDD)", value="")

        # Validate date format
        def is_valid_yyyymmdd(date_text: str) -> bool:
            try:
                datetime.strptime(date_text, '%Y%m%d')
                return True
            except ValueError:
                return False

        processed_df = None

        if st.button("Process Single File"):
            if uploaded_file is not None:
                if date_str and not is_valid_yyyymmdd(date_str):
                    st.error("Invalid date format. Please use YYYYMMDD.")
                else:
                    # Define input and output paths directly
                    SAVE_DIR = r"D:\NNE_strategy\nne_strategy\data\preprocess_trend_data"
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    
                    output_file = os.path.join(SAVE_DIR, 
                        f"trend_analysis_pp_NNE_{date_str or datetime.now().strftime('%Y%m%d')}.csv")

                    # Process the data
                    with st.spinner("Analyzing data..."):
                        try:
                            # Read the uploaded file into a DataFrame
                            df = pd.read_csv(uploaded_file)
                            df['Datetime'] = pd.to_datetime(df['Datetime'])
                            
                            # Add Action column based on trend changes
                            df['Action'] = 'Normal'
                            df.loc[df['Trend'] != df['Trend'].shift(), 'Action'] = 'Countermove'
                            
                            # Process the DataFrame
                            segments = find_countermoves(df)
                            processed_df = analyze_countermove_segments(df, segments)
                            
                            # Save the processed data
                            processed_df.to_csv(output_file, index=False)
                            st.success(f"Analysis complete. Results saved to {output_file}")
                            
                        except Exception as e:
                            st.error(f"Error analyzing file: {str(e)}")
                            st.error("Please ensure the input file contains required columns: Datetime, Close, Trend")

        # Display and visualization code
        if processed_df is not None:
            st.write("Processed Data:")
            st.dataframe(processed_df)
            
            # Add basic statistics
            st.subheader("Analysis Statistics")
            st.write(f"Total segments analyzed: {len(processed_df)}")
            if len(processed_df) > 0:
                st.write(f"Average duration: {processed_df['Duration'].mean():.2f} minutes")
                st.write(f"Average price action: {processed_df['PriceAction'].mean():.4f}")
            
    else:
        # Batch processing code remains the same
        st.subheader("Batch Processing Mode")
        input_dir = st.text_input("Input Directory", 
            value=r"D:\NNE_strategy\nne_strategy\data\preprocess_trend_data")
        output_dir = st.text_input("Output Directory", 
            value=r"D:\NNE_strategy\nne_strategy\data\counter_riversal_analysis")

        if st.button("Start Batch Processing"):
            if not os.path.isdir(input_dir):
                st.error(f"Input directory not found: {input_dir}")
            else:
                with st.spinner("Processing all files..."):
                    try:
                        files_processed = process_all_files(input_dir, output_dir)
                        st.success(f"Batch processing completed. Total files processed: {files_processed}")
                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")


###############################################################################
# MAIN APP
###############################################################################

def main():
    """Main entry point for the Streamlit UI."""
    st.set_page_config(
        page_title="Backtesting Review & Analysis",
        layout="wide"
    )
    st.divider()
    # Navigation sidebar
    st.sidebar.title("Trader's Lab ")
    screen_selection = st.sidebar.radio(
        "Select Screen",
        [
            "Backtesting Analysis",
            "Data Processing",
            "Trend Analysis",
            "Reversal Detection"  # NEW SCREEN
        ]
    )

    if screen_selection == "Backtesting Analysis":
        show_backtesting_report_screen()
    elif screen_selection == "Data Processing":
        show_data_processing_screen()
    elif screen_selection == "Trend Analysis":
        show_trend_analysis_screen()
    elif screen_selection == "Reversal Detection":
        show_countermoves_reversal_analysis_screen()
    else:
        st.write("Please select a screen from the sidebar.")


if __name__ == "__main__":
    main() 