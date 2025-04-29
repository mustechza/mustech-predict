import streamlit as st
import pandas as pd
import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import plotly.graph_objs as go

st.set_page_config(page_title="Binance Backtester", layout="wide") st.title("Binance-Backed Trading Signal Backtester")

--- User API Key Input ---

st.sidebar.header("Binance API Credentials") api_key = st.sidebar.text_input("API Key", type="password") api_secret = st.sidebar.text_input("API Secret", type="password")

client = None if api_key and api_secret: try: client = Client(api_key=api_key, api_secret=api_secret) client.get_exchange_info()  # Validate connection st.sidebar.success("Connected to Binance API") except BinanceAPIException as e: st.sidebar.error(f"Binance API error: {e.message}") except Exception as e: st.sidebar.error(f"Connection failed: {str(e)}") else: st.sidebar.warning("Enter your API credentials")

--- Asset & Interval Selection ---

if client: symbol = st.selectbox("Select Asset", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]) interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"]) limit = st.slider("Number of Candles", 100, 1000, 500)

if st.button("Fetch Historical Data"):
    try:
        raw_data = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(raw_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        st.success(f"Fetched {len(df)} candles for {symbol}")
        st.dataframe(df.tail(10))

        # --- Chart ---
        fig = go.Figure(data=[
            go.Candlestick(x=df.index,
                           open=df['open'], high=df['high'],
                           low=df['low'], close=df['close'])
        ])
        fig.update_layout(title=f"{symbol} Price Chart ({interval})", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to fetch or process data: {e}")

Footer

st.markdown("---") st.markdown("Developed for backtesting strategies with Binance historical data.")

