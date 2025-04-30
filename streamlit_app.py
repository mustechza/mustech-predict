import streamlit as st
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import datetime
from datetime import datetime

# --- API Key (Hardcoded for now) ---
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key_here"

# --- App Config ---
st.set_page_config(page_title="Breakout Signal Dashboard", layout="wide")
st.title("📈 Real-Time Breakout Signal Dashboard")

symbol = st.sidebar.selectbox("Select Symbol", ["BTCUSD", "ETHUSD", "AAPL", "GOOG"])
interval = st.sidebar.selectbox("Candle Interval", ["1min", "5min", "15min", "1h", "4h", "1d"])
limit = st.sidebar.slider("Candles to Fetch", 100, 1000, 500)

TP_PCT = st.sidebar.number_input("Take Profit (%)", 0.1, 10.0, 2.0) / 100
SL_PCT = st.sidebar.number_input("Stop Loss (%)", 0.1, 10.0, 1.0) / 100
sr_length = st.sidebar.number_input("S/R Detection Length", 5, 50, 15)
sr_margin = st.sidebar.number_input("S/R Margin", 1.0, 5.0, 2.0)
indicator_length = st.sidebar.slider("Indicator Length", 10, 50, 17)

# --- Fetch Data from Alpha Vantage ---
def fetch_alpha_vantage_data(symbol, interval, api_key, limit=500):
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series' in data:
            timeseries_key = f"Time Series ({interval})"
            df = pd.DataFrame.from_dict(data[timeseries_key], orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            return df.tail(limit)
        else:
            st.error("Failed to fetch data. Check the API key or symbol.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Fetch data for selected symbol and interval
df = fetch_alpha_vantage_data(symbol, interval, ALPHA_VANTAGE_API_KEY, limit)

# --- Indicators ---
df['ATR'] = df.ta.atr(length=indicator_length)
df['vol_sma'] = df['volume'].rolling(indicator_length).mean()

# --- Pivot Zones ---
df['pivothigh'] = df['high'][(df['high'].shift(sr_length) < df['high']) & (df['high'].shift(-sr_length) < df['high'])]
df['pivotlow'] = df['low'][(df['low'].shift(sr_length) < df['low']) & (df['low'].shift(-sr_length) < df['low'])]

zone_range = (df['high'].max() - df['low'].min()) / df['high'].max()
resistance_zones, support_zones = [], []

for i in range(sr_length, len(df) - sr_length):
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

# --- Redundant Signal Filter (Cooldown of 5 candles) ---
cooldown = 5
df['signal'] = np.where(df['bull_breakout'], 'Buy', np.where(df['bear_breakout'], 'Sell', None))
for i in range(1, len(df)):
    if df['signal'].iloc[i] == df['signal'].iloc[i - 1] and df['signal'].iloc[i] is not None:
        if i - df['signal'].last_valid_index() < cooldown:
            df.at[df.index[i], 'signal'] = None

# --- Latest Signal ---
latest = df.iloc[-1]
signal = "🟡 No Signal"
if latest['signal'] == 'Buy':
    signal = f"📈 Buy Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.2f}"
elif latest['signal'] == 'Sell':
    signal = f"📉 Sell Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.2f}"

st.subheader("🧠 Latest Signal:")
st.info(signal)

# --- Chart ---
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name='Candles'))
fig.add_trace(go.Scatter(x=df.index, y=df['resistance'], mode='lines', name='Resistance', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=df['support'], mode='lines', name='Support', line=dict(color='green')))
st.plotly_chart(fig, use_container_width=True)

# --- Signal Table & Download ---
signals_df = df[df['signal'].notna()].copy()
signals_df['Type'] = signals_df['signal']
signals_df = signals_df[['close', 'resistance', 'support', 'Type']]
signals_df.columns = ['Price', 'Resistance', 'Support', 'Type']
st.subheader("📋 Breakout Signal Log")
st.dataframe(signals_df.tail(20))
st.download_button("⬇️ Download Signal Log", signals_df.to_csv().encode('utf-8'), "signal_log.csv", "text/csv")

# --- Simple Backtest ---
def backtest_signals(data, tp_pct, sl_pct, lookahead=5):
    wins = 0
    losses = 0
    total = 0
    for i, row in data.iterrows():
        entry_price = row['Price']
        end_idx = df.index.get_loc(i) + lookahead
        if end_idx >= len(df):
            continue
        future_prices = df.iloc[df.index.get_loc(i):end_idx]['close']
        tp_price = entry_price * (1 + tp_pct) if row['Type'] == 'Buy' else entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 - sl_pct) if row['Type'] == 'Buy' else entry_price * (1 + sl_pct)
        hit_tp = (future_prices >= tp_price).any() if row['Type'] == 'Buy' else (future_prices <= tp_price).any()
        hit_sl = (future_prices <= sl_price).any() if row['Type'] == 'Buy' else (future_prices >= sl_price).any()
        if hit_tp:
            wins += 1
        elif hit_sl:
            losses += 1
        total += 1
    return wins, losses, total

wins, losses, total = backtest_signals(signals_df, TP_PCT, SL_PCT)
if total > 0:
    st.metric("Backtest Win Rate", f"{(wins/total)*100:.2f}% ({wins}/{total})")
    st.metric("Losses", f"{losses}")
else:
    st.info("Not enough signals for backtest.")
