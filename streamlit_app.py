import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import plotly.graph_objects as go

# --- Binance Setup ---
API_KEY = 'JlWm8rVqQBu7oWTBYvMZexkZWQ2uDzyXy5rEDBIvEOS8hR1vPYHt3XTZQ04KwZB4'
API_SECRET = 'pQsLSn50rGz3Nxur5I9wvNKR1CHkCXFevGe2Qa7hSGu0HV8lb74r3OBQOOVrsrlb'

#---client = Client(API_KEY, API_SECRET)---
client=Client()

# --- App Config ---
st.set_page_config(page_title="Breakout Signal Dashboard", layout="wide")
st.title("📈 Real-Time Breakout Signal Dashboard")

symbol = st.sidebar.selectbox("Select Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
interval = st.sidebar.selectbox("Candle Interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
limit = st.sidebar.slider("Candles to Fetch", 100, 1000, 500)

TP_PCT = st.sidebar.number_input("Take Profit (%)", 0.1, 10.0, 2.0) / 100
SL_PCT = st.sidebar.number_input("Stop Loss (%)", 0.1, 10.0, 1.0) / 100
sr_length = st.sidebar.number_input("S/R Detection Length", 5, 50, 15)
sr_margin = st.sidebar.number_input("S/R Margin", 1.0, 5.0, 2.0)

# --- Fetch Data ---
@st.cache_data(ttl=60)
def get_binance_ohlc(symbol, interval, limit):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except BinanceAPIException as e:
        st.error(f"Binance error: {e}")
        return pd.DataFrame()

df = get_binance_ohlc(symbol, interval, limit)

# --- Indicators ---
adj_len = sr_length
df['ATR'] = df.ta.atr(length=17)
df['vol_sma'] = df['volume'].rolling(17).mean()

# --- Pivot Zones ---
df['pivothigh'] = df['high'][(df['high'].shift(adj_len) < df['high']) & (df['high'].shift(-adj_len) < df['high'])]
df['pivotlow'] = df['low'][(df['low'].shift(adj_len) < df['low']) & (df['low'].shift(-adj_len) < df['low'])]

zone_range = (df['high'].max() - df['low'].min()) / df['high'].max()
resistance_zones, support_zones = [], []

for i in range(adj_len, len(df) - adj_len):
    idx = df.index[i]
    row = df.iloc[i]
    if not np.isnan(row['pivothigh']):
        top = row['pivothigh']
        bottom = top * (1 - sr_margin * 0.17 * zone_range)
        resistance_zones.append({'index': idx, 'top': top, 'bottom': bottom})
    if not np.isnan(row['pivotlow']):
        bottom = row['pivotlow']
        top = bottom * (1 + sr_margin * 0.17 * zone_range)
        support_zones.append({'index': idx, 'top': top, 'bottom': bottom})

df['resistance'] = np.nan
df['support'] = np.nan
for zone in resistance_zones:
    df.loc[df.index >= zone['index'], 'resistance'] = zone['top']
for zone in support_zones:
    df.loc[df.index >= zone['index'], 'support'] = zone['bottom']

# --- Signal Logic ---
df['bull_breakout'] = (df['close'] > df['resistance'].shift(1)) & (df['close'].shift(1) <= df['resistance'].shift(1))
df['bear_breakout'] = (df['close'] < df['support'].shift(1)) & (df['close'].shift(1) >= df['support'].shift(1))

# --- Last Signal ---
latest = df.iloc[-1]
signal = None
if latest['bull_breakout']:
    signal = f"📈 Buy Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.2f}"
elif latest['bear_breakout']:
    signal = f"📉 Sell Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.2f}"
else:
    signal = "🟡 No Signal"

st.subheader("🧠 Latest Signal:")
st.info(signal)

# --- Chart ---
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name='Candles'
))
fig.add_trace(go.Scatter(x=df.index, y=df['resistance'], mode='lines', name='Resistance', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=df['support'], mode='lines', name='Support', line=dict(color='green')))

st.plotly_chart(fig, use_container_width=True)

# --- Signal Table ---
signals_df = df[df['bull_breakout'] | df['bear_breakout']].copy()
signals_df['type'] = np.where(signals_df['bull_breakout'], 'Buy', 'Sell')
signals_df = signals_df[['close', 'resistance', 'support', 'type']]
signals_df.columns = ['Price', 'Resistance', 'Support', 'Type']
st.subheader("📋 Breakout Signal Log")
st.dataframe(signals_df.tail(20))
