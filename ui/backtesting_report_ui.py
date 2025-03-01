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
    group_and_analyze_countermoves,
    categorize_countermoves,
    save_analysis_results
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

def create_analysis_chart(df):
    """
    Creates an interactive chart showing price action, trends, reversals and countermoves.
    Distinguishes between uptrend and downtrend countermoves.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [Datetime, Open, High, Low, Close, Trend]
                           and optionally [min, max]
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price Action'
        )
    )

    # Add markers for minimum points (if they exist)
    min_points = df[pd.notna(df['min'])]
    if not min_points.empty:
        fig.add_trace(
            go.Scatter(
                x=min_points['Datetime'],
                y=min_points['min'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='red',
                    line=dict(width=2, color='black')
                ),
                name='Minimum Points'
            )
        )

    # Add markers for maximum points (if they exist)
    max_points = df[pd.notna(df['max'])]
    if not max_points.empty:
        fig.add_trace(
            go.Scatter(
                x=max_points['Datetime'],
                y=max_points['max'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='green',
                    line=dict(width=2, color='black')
                ),
                name='Maximum Points'
            )
        )

    # Add trend indicators if Trend column exists
    if 'Trend' in df.columns:
        # Add markers for uptrend
        uptrend = df[df['Trend'] == 'UpTrend']
        if not uptrend.empty:
            fig.add_trace(
                go.Scatter(
                    x=uptrend['Datetime'],
                    y=uptrend['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='green',
                        opacity=0.5
                    ),
                    name='UpTrend'
                )
            )

        # Add markers for downtrend
        downtrend = df[df['Trend'] == 'DownTrend']
        if not downtrend.empty:
            fig.add_trace(
                go.Scatter(
                    x=downtrend['Datetime'],
                    y=downtrend['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=8,
                        color='red',
                        opacity=0.5
                    ),
                    name='DownTrend'
                )
            )

    # Update layout
    fig.update_layout(
        title='Price Action with Trends and Key Points',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


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

def show_trend_analysis_screen():
    """Displays the Trend Analysis UI with manual point creation functionality."""
    st.header("Trend Analysis")

    # Step 1: File Upload
    st.subheader("1️⃣ Upload Raw Stock Data")
    uploaded_file = st.file_uploader("Upload a CSV file containing stock data", type="csv")

    # Only proceed with analysis if a file is uploaded
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            
            # Display the first few rows to help diagnose issues
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Map columns from the raw file format to the expected format
            column_mapping = {
                'date': 'Datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Check if the file has the expected raw format columns
            raw_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in raw_columns):
                # Rename columns to match expected format
                df = df.rename(columns=column_mapping)
                st.success("Successfully mapped columns from raw format to expected format")
            else:
                # Check for required columns in the expected format
                required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.write("Available columns:", ", ".join(df.columns))
                    
                    # Allow manual column mapping
                    st.subheader("Manual Column Mapping")
                    st.write("Please map the available columns to the required columns:")
                    
                    mapping = {}
                    for required_col in required_columns:
                        mapping[required_col] = st.selectbox(
                            f"Select column to use as '{required_col}':",
                            options=[""] + list(df.columns),
                            key=f"map_{required_col}"
                        )
                    
                    if st.button("Apply Mapping"):
                        # Check if all required columns are mapped
                        if all(mapping.values()):
                            # Create a new DataFrame with the mapped columns
                            mapped_df = pd.DataFrame()
                            for required_col, source_col in mapping.items():
                                mapped_df[required_col] = df[source_col]
                            
                            # Replace the original DataFrame with the mapped one
                            df = mapped_df
                            st.success("Column mapping applied successfully")
                        else:
                            st.error("Please map all required columns")
                            return
                    else:
                        return
            
            # Convert Datetime column to datetime type
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Initialize empty points DataFrame for manual editing if not already in session state
            if 'edited_points' not in st.session_state:
                st.session_state.edited_points = pd.DataFrame(columns=['Datetime', 'min', 'max'])
            
            # Step 2: Manual Point Creation
            st.subheader("2️⃣ Manual Point Creation")
            st.write("Click on the chart to add minimum and maximum points. You can delete points if needed.")
            
            try:
                # Import plotly_events for interactive editing
                from streamlit_plotly_events import plotly_events
                
                # Create function to generate the interactive chart
                def create_interactive_chart(df, points_df):
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(
                        go.Scatter(
                            x=df['Datetime'],
                            y=df['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue', width=1)
                        )
                    )
                    
                    # Add minima points
                    min_points = points_df[pd.notna(points_df['min'])]
                    if not min_points.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=min_points['Datetime'],
                                y=min_points['min'],
                                mode='markers',
                                name='Minima',
                                marker=dict(
                                    color='red',
                                    size=10,
                                    symbol='circle'
                                ),
                                customdata=min_points.index.tolist(),
                                hovertemplate='Time: %{x}<br>Price: %{y}<br>Index: %{customdata}<extra></extra>'
                            )
                        )
                    
                    # Add maxima points
                    max_points = points_df[pd.notna(points_df['max'])]
                    if not max_points.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=max_points['Datetime'],
                                y=max_points['max'],
                                mode='markers',
                                name='Maxima',
                                marker=dict(
                                    color='green',
                                    size=10,
                                    symbol='circle'
                                ),
                                customdata=max_points.index.tolist(),
                                hovertemplate='Time: %{x}<br>Price: %{y}<br>Index: %{customdata}<extra></extra>'
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title='Interactive Price Chart - Click to Add Points',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        height=600,
                        hovermode='closest',
                        dragmode='pan'  # Set dragmode to 'pan' for better navigation
                    )
                    
                    # Add configuration for better user experience
                    fig.update_layout(
                        modebar=dict(
                            orientation='v',
                            remove=[]
                        )
                    )
                    
                    return fig
                
                # Set up edit mode selection
                edit_mode = st.radio(
                    "Edit Mode:",
                    ["View Only", "Add Min Point", "Add Max Point", "Delete Point"],
                    horizontal=True,
                    key="edit_mode_select"
                )
                
                # Create the interactive chart
                fig = create_interactive_chart(df, st.session_state.edited_points)
                
                # Only capture and process events if we're not in view-only mode
                if edit_mode != "View Only":
                    # Display the chart and capture click events
                    selected_points = plotly_events(
                        fig, 
                        click_event=True,
                        select_event=False,
                        override_height=600,
                        override_width="100%",
                        key=f"interactive_chart_{edit_mode}"
                    )
                    
                    # Process click events only when points are actually selected
                    if selected_points and len(selected_points) > 0:
                        point = selected_points[0]
                        x_val = pd.to_datetime(point['x'])
                        y_val = point['y']
                        
                        # Fix timezone issue
                        if df['Datetime'].dt.tz is not None:
                            if x_val.tz is None:
                                # Add timezone to match df
                                x_val = x_val.tz_localize(df['Datetime'].dt.tz)
                        else:
                            if x_val.tz is not None:
                                # Remove timezone from x_val
                                x_val = x_val.tz_localize(None)
                        
                        # Find the closest time in the original dataframe
                        closest_idx = None
                        min_diff = pd.Timedelta.max
                        
                        for idx, row_time in enumerate(df['Datetime']):
                            # Ensure both times are comparable
                            if row_time.tz != x_val.tz:
                                if row_time.tz is None:
                                    compare_time = row_time
                                else:
                                    compare_time = row_time.tz_localize(None)
                                    
                                if x_val.tz is None:
                                    compare_x = x_val
                                else:
                                    compare_x = x_val.tz_localize(None)
                            else:
                                compare_time = row_time
                                compare_x = x_val
                            
                            diff = abs(compare_time - compare_x)
                            if diff < min_diff:
                                min_diff = diff
                                closest_idx = idx
                        
                        if closest_idx is not None:
                            closest_time = df.loc[closest_idx, 'Datetime']
                            y_val = df.loc[closest_idx, 'Close']  # Use the actual close price at that time
                        else:
                            st.warning("Could not find a matching time point in the data.")
                            closest_time = x_val
                        
                        # Handle different edit modes
                        if edit_mode == "Add Min Point":
                            # Check if a point already exists at this time
                            existing = st.session_state.edited_points[
                                st.session_state.edited_points['Datetime'] == closest_time
                            ]
                            
                            if not existing.empty:
                                # Update existing point
                                idx = existing.index[0]
                                st.session_state.edited_points.loc[idx, 'min'] = y_val
                                st.session_state.edited_points.loc[idx, 'max'] = None
                                st.success(f"Updated point at {closest_time.strftime('%H:%M:%S')} to minimum")
                            else:
                                # Add new point
                                new_point = pd.DataFrame({
                                    'Datetime': [closest_time],
                                    'min': [y_val],
                                    'max': [None]
                                })
                                st.session_state.edited_points = pd.concat([st.session_state.edited_points, new_point])
                                st.success(f"Added minimum point at {closest_time.strftime('%H:%M:%S')}")
                            
                            # Sort points by datetime
                            st.session_state.edited_points = st.session_state.edited_points.sort_values('Datetime').reset_index(drop=True)
                            
                        elif edit_mode == "Add Max Point":
                            # Check if a point already exists at this time
                            existing = st.session_state.edited_points[
                                st.session_state.edited_points['Datetime'] == closest_time
                            ]
                            
                            if not existing.empty:
                                # Update existing point
                                idx = existing.index[0]
                                st.session_state.edited_points.loc[idx, 'max'] = y_val
                                st.session_state.edited_points.loc[idx, 'min'] = None
                                st.success(f"Updated point at {closest_time.strftime('%H:%M:%S')} to maximum")
                            else:
                                # Add new point
                                new_point = pd.DataFrame({
                                    'Datetime': [closest_time],
                                    'min': [None],
                                    'max': [y_val]
                                })
                                st.session_state.edited_points = pd.concat([st.session_state.edited_points, new_point])
                                st.success(f"Added maximum point at {closest_time.strftime('%H:%M:%S')}")
                            
                            # Sort points by datetime
                            st.session_state.edited_points = st.session_state.edited_points.sort_values('Datetime').reset_index(drop=True)
                            
                        elif edit_mode == "Delete Point":
                            # Delete the selected point
                            if 'customdata' in point:
                                idx = point['customdata']
                                if idx < len(st.session_state.edited_points):
                                    point_time = st.session_state.edited_points.loc[idx, 'Datetime']
                                    st.session_state.edited_points = st.session_state.edited_points.drop(idx).reset_index(drop=True)
                                    st.success(f"Deleted point at {point_time.strftime('%H:%M:%S')}")

                        # Show updated points
                        st.write("Current Points:")
                        st.dataframe(st.session_state.edited_points[['Datetime', 'min', 'max']].sort_values('Datetime'))
                else:
                    # In view-only mode, just display the chart without event handling
                    st.plotly_chart(fig, use_container_width=True)

                # Add helper text
                st.markdown("""
                ### How to use:
                - **View Only**: See your current points without making changes
                - **Add Min Point**: Click anywhere on the chart to add a minimum point (red)
                - **Add Max Point**: Click anywhere on the chart to add a maximum point (green)
                - **Delete Point**: Click on an existing point to remove it
                
                For optimal trend analysis, ensure that minimum and maximum points alternate.
                """)

            except ImportError:
                st.error("The streamlit-plotly-events package is required for interactive editing.")
                st.code("pip install streamlit-plotly-events", language="bash")
                st.warning("Please install the package and restart the application.")
            except Exception as e:
                st.error(f"An unexpected error occurred in the interactive chart: {str(e)}")
                st.exception(e)
            
            # Step 3: Generate Trend Analysis
            st.subheader("3️⃣ Generate Trend Analysis")
            
            if st.button("Generate Trend Analysis"):
                if len(st.session_state.edited_points) < 2:
                    st.error("You need at least 2 points to analyze trends.")
                    return
                
                # Ensure points alternate between min and max
                points_df = st.session_state.edited_points.copy().sort_values('Datetime').reset_index(drop=True)
                
                # Analyze trends
                trends = []
                for i in range(len(points_df) - 1):
                    start = points_df.iloc[i]
                    end = points_df.iloc[i + 1]
                    
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
                        'Start Time': start['Datetime'],
                        'Start Price': start_price,
                        'End Time': end['Datetime'],
                        'End Price': end_price
                    })
                
                processed_df = df.copy()
                processed_df['min'] = None
                processed_df['max'] = None
                processed_df['Trend'] = None
                
                # Add points to processed dataframe
                for _, row in points_df.iterrows():
                    # Find the matching datetime
                    match_idx = processed_df[processed_df['Datetime'] == row['Datetime']].index
                    if len(match_idx) > 0:
                        if pd.notna(row['min']):
                            processed_df.loc[match_idx[0], 'min'] = row['min']
                        if pd.notna(row['max']):
                            processed_df.loc[match_idx[0], 'max'] = row['max']
                
                # Apply trends to the dataframe
                for trend in trends:
                    mask = (processed_df['Datetime'] >= trend['Start Time']) & (processed_df['Datetime'] <= trend['End Time'])
                    processed_df.loc[mask, 'Trend'] = trend['Trend Type']
                
                # Create analysis chart
                fig = create_analysis_chart(processed_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trend summary
                st.subheader("Trend Summary")
                if trends:
                    trend_df = pd.DataFrame(trends)
                    trend_df['Duration (min)'] = (trend_df['End Time'] - trend_df['Start Time']).dt.total_seconds() / 60
                    trend_df['Price Change'] = trend_df['End Price'] - trend_df['Start Price']
                    trend_df['% Change'] = (trend_df['Price Change'] / trend_df['Start Price']) * 100
                    st.dataframe(trend_df)
                else:
                    st.warning("No valid trends identified. Please ensure you have alternating min/max points.")

                # (NEW) Store processed_df in session state so it stays available
                st.session_state.processed_df = processed_df
                st.success("Trend analysis generated. You can now proceed to Save Analysis.")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("Please upload a CSV file to begin analysis.")
        return
    
    # Step 4: Save Analysis
    st.subheader("4️⃣ Save Analysis")
    col1, col2 = st.columns(2)
    with col1:
        stock_name = st.text_input("Enter Stock Symbol:", value="STOCK")
        # Ensure stock name is valid for filenames
        stock_name = re.sub(r'[^\w\-]', '_', stock_name)
    with col2:
        # We'll fetch a date if possible, else just use today's date
        if 'processed_df' in st.session_state and not st.session_state.processed_df.empty:
            file_date = st.session_state.processed_df['Datetime'].iloc[0].strftime('%Y-%m-%d')
        else:
            file_date = date.today().strftime("%Y-%m-%d")
        st.write(f"Analysis Date: {file_date}")

    # Create filename in required format
    save_name = f"trend_{stock_name}_{file_date}"

    # Add option to choose save location
    use_default_folder = st.checkbox("Use default save location", value=True)

    if not use_default_folder:
        custom_folder = st.text_input("Enter custom save folder path:")
        save_folder = custom_folder if custom_folder else TREND_FOLDER
    else:
        save_folder = TREND_FOLDER

    if st.button("Save Analysis"):
        # Retrieve the df from session state
        processed_df = st.session_state.get("processed_df")

        # Basic check to prevent crashes
        if processed_df is None or processed_df.empty:
            st.error("No data to save. Please generate trend analysis first.")
            return

        try:
            # Diagnostic info
            st.info("Debug info:")
            st.write(f"- Directory exists: {os.path.exists(save_folder)}")
            st.write(f"- Directory writable: {os.access(save_folder, os.W_OK)}")
            st.write(f"- DataFrame shape: {processed_df.shape}")
            st.write(f"- DataFrame columns: {processed_df.columns.tolist()}")

            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{save_name}.csv")
            st.write(f"- Full save path: {save_path}")

            # Convert df to CSV in memory
            buffer = io.StringIO()
            processed_df.to_csv(buffer, index=False)
            buffer.seek(0)
            csv_data = buffer.getvalue()
            st.write(f"- DataFrame successfully converted to CSV (size: {len(csv_data)} bytes)")

            # Write it to disk
            with open(save_path, 'w', newline='') as f:
                f.write(csv_data)

            # Verify creation
            if os.path.exists(save_path):
                st.success(f"Analysis saved successfully to:\n{save_path}")
                # Provide a download button
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{save_name}.csv",
                    mime='text/csv'
                )
            else:
                st.error(f"File was not created at {save_path} despite no errors.")
                st.warning("You can still download the file below:")
                st.download_button(
                    label="Download CSV directly",
                    data=csv_data,
                    file_name=f"{save_name}.csv",
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Error saving analysis: {str(e)}")
            st.exception(e)
            # Fallback: allow direct download
            try:
                buffer = io.StringIO()
                processed_df.to_csv(buffer, index=False)
                buffer.seek(0)
                st.warning("File couldn't be saved locally, but you can download it:")
                st.download_button(
                    label="Download CSV (recovery)",
                    data=buffer.getvalue(),
                    file_name=f"{save_name}.csv",
                    mime='text/csv'
                )
            except Exception as download_err:
                st.error(f"Also failed to provide download: {str(download_err)}")


###############################################################################
# SCREEN: REVERSAL ANALYSIS
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
        uploaded_file = st.file_uploader("Upload a CSV file containing trend analysis data", type="csv")

        if st.button("Process Single File"):
            if uploaded_file is not None:
                try:
                    # Define input and output paths directly
                    SAVE_DIR = r"D:\NNE_strategy\nne_strategy\data\preprocess_trend_data"
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    
                    # Read the uploaded file into a DataFrame
                    df = pd.read_csv(uploaded_file)
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    
                    # Verify all required columns
                    required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'min', 'max', 'Trend']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        st.error(f"Missing required columns: {missing_columns}")
                        return
                    
                    # Extract date from the data for filename
                    date_str = df['Datetime'].iloc[0].strftime('%Y%m%d')
                    
                    # Save the input DataFrame to a temporary file
                    temp_input = os.path.join(SAVE_DIR, f"temp_input_{date_str}.csv")
                    df.to_csv(temp_input, index=False)
                    
                    # Define output file path
                    output_file = os.path.join(SAVE_DIR, f"trend_analysis_pp_NNE_{date_str}.csv")
                    
                    # Use analyze_trends with the temporary file
                    analyze_trends(temp_input, output_file)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_input):
                        os.remove(temp_input)
                    
                    st.success(f"Analysis complete. Results saved to {output_file}")
                    
                    # Show preview of results with visualization
                    try:
                        result_df = pd.read_csv(output_file)
                        result_df['Datetime'] = pd.to_datetime(result_df['Datetime'])
                        
                        # Create tabs for different views
                        st.write("Analysis Results:")
                        tabs = st.tabs(["Chart", "Statistics", "Data"])
                        
                        # Chart tab
                        with tabs[0]:
                            st.subheader("Visual Analysis")
                            fig = create_analysis_chart(result_df)
                            st.plotly_chart(fig, use_container_width=True)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Reversals", 
                                        len(result_df[result_df['Action'] == 'Reversal']))
                            with col2:
                                st.metric("Uptrend Countermoves", 
                                        len(result_df[(result_df['Action'] == 'Countermove') & 
                                                    (result_df['Trend'] == 'UpTrend')]))
                            with col3:
                                st.metric("Downtrend Countermoves", 
                                        len(result_df[(result_df['Action'] == 'Countermove') & 
                                                    (result_df['Trend'] == 'DownTrend')]))
                        
                        # Statistics tab
                        with tabs[1]:
                            st.subheader("Statistical Analysis")
                            show_statistics_tab(result_df)
                        
                        # Data tab
                        with tabs[2]:
                            st.subheader("Data Preview")
                            st.write(result_df)
                            
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                        st.exception(e)
                
                except Exception as e:
                    st.error(f"Error analyzing file: {str(e)}")
                    st.error("Please ensure the input file contains all required columns")

        else:
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
# STATISTICS FUNCTIONS
###############################################################################

def analyze_statistics(df):
    """
    Analyze countermoves and reversals statistics using the core analysis functions.
    Returns:
        tuple: (pos_grouped_stats, neg_grouped_stats, reversals)
    """
    df = df.copy()
    is_streamlit_context = True
    try:
        _ = st.session_state  # Will raise exception if outside Streamlit context
    except:
        is_streamlit_context = False

    # Ensure 'Action' is present, run find_countermoves if needed.
    if 'Action' not in df.columns:
        df = find_countermoves(df)

    # Use the new grouping logic instead of the old row-based segments:
    grouped_df = group_and_analyze_countermoves(df)
    
    # Fix column naming inconsistencies
    
    # Map PriceChangePct to PricePct
    if 'PriceChangePct' in grouped_df.columns and 'PricePct' not in grouped_df.columns:
        grouped_df['PricePct'] = grouped_df['PriceChangePct']
        if is_streamlit_context:
            st.info("Mapped 'PriceChangePct' to 'PricePct' for compatibility")
    
    # Add PriceAction column if missing - should be absolute price change
    if 'PriceAction' not in grouped_df.columns:
        # Use PriceChange as PriceAction if available (absolute, not percentage)
        if 'PriceChange' in grouped_df.columns:
            grouped_df['PriceAction'] = grouped_df['PriceChange']
            if is_streamlit_context:
                st.info("Mapped 'PriceChange' to 'PriceAction' for categorization")
        else:
            # Calculate absolute price change if needed
            if 'StartPrice' in grouped_df.columns and 'EndPrice' in grouped_df.columns:
                grouped_df['PriceAction'] = grouped_df.apply(
                    lambda row: abs(row['EndPrice'] - row['StartPrice']), axis=1)
                if is_streamlit_context:
                    st.info("Calculated 'PriceAction' as absolute price change")
            else:
                # Last resort - use whatever price data we can find
                if is_streamlit_context:
                    st.warning("Missing price data for 'PriceAction' calculation")
                grouped_df['PriceAction'] = 0
    
    # Handle column mapping: check for 'Direction' and if not present, 
    # try to map from 'Trend' column
    if 'Direction' not in grouped_df.columns:
        if 'Trend' in grouped_df.columns:
            # Map from Trend to Direction
            trend_direction_map = {'UpTrend': 'positive', 'DownTrend': 'negative'}
            grouped_df['Direction'] = grouped_df['Trend'].map(trend_direction_map)
            if is_streamlit_context:
                st.info("Mapped 'Trend' column to 'Direction' for analysis")
        else:
            # Create Direction based on price movement if neither column exists
            if is_streamlit_context:
                st.warning("Neither 'Direction' nor 'Trend' columns found. Creating direction based on price movement.")
            if 'PricePct' in grouped_df.columns:
                grouped_df['Direction'] = grouped_df['PricePct'].apply(
                    lambda x: 'positive' if x >= 0 else 'negative')
            elif all(col in grouped_df.columns for col in ['StartPrice', 'EndPrice']):
                grouped_df['Direction'] = grouped_df.apply(
                    lambda row: 'positive' if row['EndPrice'] >= row['StartPrice'] 
                                else 'negative', axis=1)
            else:
                st.error("Cannot determine direction: missing required price columns")
                return pd.DataFrame(), pd.DataFrame(), df[df['Action'] == 'Reversal'].copy()

    # Split positive vs. negative
    try:
        pos_df = grouped_df[grouped_df['Direction'] == 'positive'].copy()
        neg_df = grouped_df[grouped_df['Direction'] == 'negative'].copy()
    except KeyError:
        st.error("Failed to filter by direction - check column mapping")
        return pd.DataFrame(), pd.DataFrame(), df[df['Action'] == 'Reversal'].copy()

    # Categorize
    try:
        if not pos_df.empty and 'PricePct' in pos_df.columns and 'PriceAction' in pos_df.columns:
            _, pos_grouped_stats = categorize_countermoves(pos_df, group_by='PricePct')
        else:
            if pos_df.empty:
                if is_streamlit_context:
                    st.info("No positive countermoves found")
            else:
                if is_streamlit_context:
                    st.warning(f"Missing required columns in positive data. Available: {list(pos_df.columns)}")
            pos_grouped_stats = pd.DataFrame()
            
        if not neg_df.empty and 'PricePct' in neg_df.columns and 'PriceAction' in neg_df.columns:
            _, neg_grouped_stats = categorize_countermoves(neg_df, group_by='PricePct')
        else:
            if neg_df.empty:
                if is_streamlit_context:
                    st.info("No negative countermoves found")
            else:
                if is_streamlit_context:
                    st.warning(f"Missing required columns in negative data. Available: {list(neg_df.columns)}")
            neg_grouped_stats = pd.DataFrame()
            
    except Exception as e:
        st.exception(e)
        pos_grouped_stats = pd.DataFrame()
        neg_grouped_stats = pd.DataFrame()

    # Also return the raw Reversal rows from df for potential display
    reversals = df[df['Action'] == 'Reversal'].copy()

    return pos_grouped_stats, neg_grouped_stats, reversals


def show_statistics_tab(result_df):
    """
    Demonstration tab for displaying the new grouped statistics.
    """
    pos_grouped_stats, neg_grouped_stats, reversals = analyze_statistics(result_df)
    
    # Summary metrics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patterns", 
                  len(result_df[result_df['Action'].isin(['Countermove', 'Reversal'])]))
    with col2:
        st.metric("Positive Countermoves", 
                  len(result_df[(result_df['Action'] == 'Countermove') & 
                                (result_df['Close'] > result_df['Close'].shift(1))]))
    with col3:
        st.metric("Negative Countermoves", 
                  len(result_df[(result_df['Action'] == 'Countermove') & 
                                (result_df['Close'] < result_df['Close'].shift(1))]))

    # Debug info for durations
    st.subheader("Debug Information - Grouped Stats")
    if isinstance(pos_grouped_stats, pd.DataFrame) and not pos_grouped_stats.empty:
        st.write("Positive Countermoves Duration Check:")
        for _, row in pos_grouped_stats.iterrows():
            st.write(f"{row['SizeGroup']}: {row['AvgDuration']:.2f} minutes")

    if isinstance(neg_grouped_stats, pd.DataFrame) and not neg_grouped_stats.empty:
        st.write("Negative Countermoves Duration Check:")
        for _, row in neg_grouped_stats.iterrows():
            st.write(f"{row['SizeGroup']}: {row['AvgDuration']:.2f} minutes")

    # Save the results
    stats_dir = r"D:\NNE_strategy\nne_strategy\data\stats"
    os.makedirs(stats_dir, exist_ok=True)
    date_str = result_df['Datetime'].iloc[0].strftime('%Y%m%d')
    stats_file = os.path.join(stats_dir, f"countermove_analysis_{date_str}.json")
    
    # Convert to DataFrames if not already
    if not isinstance(pos_grouped_stats, pd.DataFrame):
        pos_df = pd.DataFrame() if not pos_grouped_stats else pd.DataFrame(pos_grouped_stats)
    else:
        pos_df = pos_grouped_stats
        
    if not isinstance(neg_grouped_stats, pd.DataFrame):
        neg_df = pd.DataFrame() if not neg_grouped_stats else pd.DataFrame(neg_grouped_stats)
    else:
        neg_df = neg_grouped_stats
    
    try:
        save_analysis_results(pos_df, neg_df, stats_file)
        st.success(f"Statistics saved to {stats_file}")
    except Exception as e:
        st.error(f"Error saving statistics: {str(e)}")
        st.exception(e)

    # Display data frames
    st.subheader("Detailed Statistics")
    if isinstance(pos_grouped_stats, pd.DataFrame) and not pos_grouped_stats.empty:
        st.write("Positive Countermoves by Size:")
        st.dataframe(pos_grouped_stats)
    
    if isinstance(neg_grouped_stats, pd.DataFrame) and not neg_grouped_stats.empty:
        st.write("Negative Countermoves by Size:")
        st.dataframe(neg_grouped_stats)


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