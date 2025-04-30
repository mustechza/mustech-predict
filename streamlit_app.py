import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
import requests
import plotly.graph_objects as go
import time

# --- Auto-refresh every 60 sec ---
st_autorefresh = st.experimental_rerun if st.button("🔄 Refresh Now") else None
st_autorefresh = st_autorefresh or st.experimental_rerun if st.experimental_get_query_params().get("auto") else None
st_autorefresh = st_autorefresh or st.experimental_rerun if time.time() % 60 < 1 else None

# --- App Config ---
st.set_page_config(page_title="Breakout Signal Dashboard", layout="wide")
st.title("📈 Real-Time Breakout Signal Dashboard (No API Key)")

# --- Sidebar Options ---
symbol = st.sidebar.selectbox("Select Symbol", [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"
])
interval = st.sidebar.selectbox("Candle Interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
limit = st.sidebar.slider("Candles to Fetch", 100, 1000, 500)

TP_PCT = st.sidebar.number_input("Take Profit (%)", 0.1, 10.0, 2.0) / 100
SL_PCT = st.sidebar.number_input("Stop Loss (%)", 0.1, 10.0, 1.0) / 100
sr_length = st.sidebar.number_input("S/R Detection Length", 5, 50, 15)
sr_margin = st.sidebar.number_input("S/R Margin", 1.0, 5.0, 2.0)
indicator_length = st.sidebar.slider("Indicator Length", 10, 50, 17)

# --- Binance Public API Fetch ---
@st.cache_data(ttl=60)
def get_binance_ohlc(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

df = get_binance_ohlc(symbol, interval, limit)

# --- Indicators ---
adj_len = sr_length
df['ATR'] = df.ta.atr(length=indicator_length)
df['vol_sma'] = df['volume'].rolling(indicator_length).mean()

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

# --- Cooldown ---
cooldown = 5
df['signal'] = np.where(df['bull_breakout'], 'Buy', np.where(df['bear_breakout'], 'Sell', None))
last_signal_idx = -cooldown * 2
for i in range(len(df)):
    if df['signal'].iloc[i]:
        if i - last_signal_idx < cooldown:
            df.at[df.index[i], 'signal'] = None
        else:
            last_signal_idx = i

# --- Latest Signal ---
latest = df.iloc[-1]
signal = "🟡 No Signal"
if latest['signal'] == 'Buy':
    signal = f"📈 Buy Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.4f}"
elif latest['signal'] == 'Sell':
    signal = f"📉 Sell Signal ({symbol}) at {latest.name.strftime('%H:%M:%S')} - Price: {latest['close']:.4f}"

st.subheader("🧠 Latest Signal:")
st.info(signal)

# --- Chart ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name='Candles'))
fig.add_trace(go.Scatter(x=df.index, y=df['resistance'], mode='lines', name='Resistance', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=df['support'], mode='lines', name='Support', line=dict(color='green')))
st.plotly_chart(fig, use_container_width=True)

# --- Signal Table ---
signals_df = df[df['signal'].notna()].copy()
signals_df['Type'] = signals_df['signal']
signals_df = signals_df[['close', 'resistance', 'support', 'Type']]
signals_df.columns = ['Price', 'Resistance', 'Support', 'Type']
st.subheader("📋 Breakout Signal Log")
st.dataframe(signals_df.tail(20))

st.download_button("⬇️ Download Signal Log", signals_df.to_csv().encode('utf-8'), "signal_log.csv", "text/csv")

# --- Backtest ---
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
