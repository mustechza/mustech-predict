import streamlit as st
import pandas as pd
from binance.client import Client
import time

# Binance credentials (for public endpoints, no need for keys)
client = Client()

# Supported symbols
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT"]

# Function to fetch candles
@st.cache_data(show_spinner=False)
def fetch_binance_data(symbol):
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=500)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Simple EMA Crossover strategy
def ema_cross_strategy(df):
    df['EMA5'] = df['close'].ewm(span=5).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['Signal'] = 0
    df.loc[df['EMA5'] > df['EMA20'], 'Signal'] = 1
    df.loc[df['EMA5'] < df['EMA20'], 'Signal'] = -1
    return df

# Backtest with Martingale (2-step limit)
def backtest_martingale(df, bet=1.0, limit=2):
    balance = 100
    position = 0
    steps = 0
    history = []

    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i-1]
        outcome = 1 if df['close'].iloc[i] > df['close'].iloc[i-1] else -1

        if signal == 1:  # Buy signal
            profit = bet * outcome
            balance += profit
            history.append((df.index[i], bet, profit, balance))
            if outcome < 0 and steps < limit:
                bet *= 2
                steps += 1
            else:
                bet = 1.0
                steps = 0

    return pd.DataFrame(history, columns=["Time", "Bet", "Profit", "Balance"])

# Sidebar controls
st.sidebar.title("Backtest Control")
use_uploaded = st.sidebar.checkbox("Use uploaded CSV", False)
selected_symbol = st.sidebar.selectbox("Select Asset", symbols)

# Upload or fetch
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
        st.success("File loaded.")
    else:
        st.stop()
else:
    with st.spinner(f"Fetching data for {selected_symbol}..."):
        df = fetch_binance_data(selected_symbol)

# Show price chart
st.subheader(f"{selected_symbol} - Last 500 candles")
st.line_chart(df['close'])

# Strategy and backtest
st.subheader("Backtest Results (Martingale)")
df = ema_cross_strategy(df)
backtest_df = backtest_martingale(df)
st.dataframe(backtest_df.tail(10))
st.line_chart(backtest_df.set_index("Time")["Balance"])

# Final metrics
st.metric("Final Balance", f"${backtest_df['Balance'].iloc[-1]:.2f}")
st.metric("Total Trades", len(backtest_df))
