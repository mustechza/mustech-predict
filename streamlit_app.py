import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# --- API Key from Streamlit Secrets ---
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]

# --- App Config ---
st.set_page_config(page_title="Breakout Signal Dashboard", layout="wide")
st.title("📈 Real-Time Breakout Signal Dashboard")

# --- Sidebar Configuration ---
symbol = st.sidebar.selectbox("Select Symbol", ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"])
interval = st.sidebar.selectbox("Candle Interval", ["1min", "5min", "15min", "1h", "4h", "1d"])
limit = st.sidebar.slider("Candles to Fetch", 100, 1000, 500)

TP_PCT = st.sidebar.number_input("Take Profit (%)", 0.1, 10.0, 2.0) / 100
SL_PCT = st.sidebar.number_input("Stop Loss (%)", 0.1, 10.0, 1.0) / 100

# --- Fetch Data from Alpha Vantage API ---
def get_alpha_vantage_data(symbol, interval, limit):
    try:
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'  # 'compact' for 100 data points, 'full' for max (over 2000 data points)
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if the data retrieval is successful
        if "Time Series" in data:
            time_series_key = f"Time Series ({interval})"
            df = pd.DataFrame(data[time_series_key]).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            return df.head(limit)
        else:
            st.error(f"Failed to fetch data for {symbol}.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()

# Fetch the data
df = get_alpha_vantage_data(symbol, interval, limit)

# --- Calculate ATR and Signal Logic ---
indicator_length = 14
df['ATR'] = df['high'].rolling(indicator_length).mean() - df['low'].rolling(indicator_length).mean()  # Simplified ATR

# --- Signal Logic ---
df['bull_breakout'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) <= df['close'].shift(1))
df['bear_breakout'] = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) >= df['close'].shift(1))

# --- Latest Signal ---
latest = df.iloc[-1]
signal = "🟡 No Signal"
if latest['bull_breakout']:
    signal = f"📈 Buy Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.2f}"
elif latest['bear_breakout']:
    signal = f"📉 Sell Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.2f}"

st.subheader("🧠 Latest Signal:")
st.info(signal)

# --- Plot Chart ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name='Candles'))
st.plotly_chart(fig, use_container_width=True)

# --- Signal Table & Download ---
signals_df = df[df['bull_breakout'] | df['bear_breakout']].copy()
signals_df['Type'] = signals_df.apply(lambda row: 'Buy' if row['bull_breakout'] else 'Sell' if row['bear_breakout'] else None, axis=1)
signals_df = signals_df[['close', 'Type']]
signals_df.columns = ['Price', 'Type']
st.subheader("📋 Breakout Signal Log")
st.dataframe(signals_df.tail(20))
st.download_button("⬇️ Download Signal Log", signals_df.to_csv().encode('utf-8'), "signal_log.csv", "text/csv")

