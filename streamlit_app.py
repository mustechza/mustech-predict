import streamlit as st 
import pandas as pd 
import datetime 
from binance.client import Client 
from binance.exceptions 
import BinanceAPIException 
import plotly.graph_objs as go

st.set_page_config(page_title="Binance Backtester", layout="wide") st.title("Binance-Backed Trading Signal Backtester")

st.sidebar.header("API Credentials") api_key = st.sidebar.text_input("API Key", type="password") api_secret = st.sidebar.text_input("API Secret", type="password")

client = None if api_key and api_secret: try: client = Client(api_key, api_secret) client.ping() st.sidebar.success("API Connected") except BinanceAPIException as e: st.sidebar.error(f"API Error: {e.message}")

st.sidebar.header("Data Parameters") symbol = st.sidebar.selectbox("Asset", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]) interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h"]) limit = st.sidebar.slider("Number of Candles", min_value=100, max_value=1000, value=500)

if client: with st.spinner("Fetching data from Binance..."): try: klines = client.get_klines(symbol=symbol, interval=interval, limit=limit) df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"]) df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms') df.set_index("timestamp", inplace=True) df = df[["open", "high", "low", "close", "volume"]].astype(float)

st.success(f"Loaded {len(df)} candles for {symbol} on {interval} interval.")

        # Display chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        ))
        st.plotly_chart(fig, use_container_width=True)

    except BinanceAPIException as e:
        st.error(f"Error fetching data: {e.message}")

else: st.warning("Please enter your Binance API key and secret to proceed.")

