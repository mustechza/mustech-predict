import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime

# Auto-refresh every 60 seconds
count = st_autorefresh(interval=60000, limit=None, key="datarefresh")

st.title("Binance Breakout Trading Dashboard")

# Sidebar inputs
api_key = st.sidebar.text_input("Binance API Key", type="password")
api_secret = st.sidebar.text_input("Binance API Secret", type="password")
symbol = st.sidebar.text_input("Symbol (e.g. BTCUSDT)", value="BTCUSDT")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h","4h"], index=3)
lookback = st.sidebar.number_input("Lookback candles", min_value=20, value=100, step=10)

if not api_key or not api_secret:
    st.warning("Please enter Binance API credentials.")
    st.stop()

# Initialize Binance client
client = Client(api_key, api_secret)

# Fetch historical klines
klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
if not klines:
    st.error("Failed to fetch data. Check symbol and API credentials.")
    st.stop()

# Prepare DataFrame
df = pd.DataFrame(klines, columns=[
    'open_time','open','high','low','close','volume','close_time',
    'quote_asset_volume','trades','taker_buy_base','taker_buy_quote','ignore'
])
df = df.astype({'open_time': int, 'open': float, 'high': float, 
                'low': float, 'close': float, 'volume': float})
df['time'] = pd.to_datetime(df['open_time'], unit='ms')
df = df[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

# Calculate ATR (Average True Range)
df['prev_close'] = df['close'].shift(1)
df['HL'] = df['high'] - df['low']
df['HC'] = (df['high'] - df['prev_close']).abs()
df['LC'] = (df['low'] - df['prev_close']).abs()
df['TR'] = df[['HL','HC','LC']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()
# ATR rolling average for filter
df['ATR_avg'] = df['ATR'].rolling(window=14).mean()

# Volume rolling average
df['vol_avg'] = df['volume'].rolling(window=14).mean()

# Define breakout zones (e.g., rolling high/low of last 20 bars)
df['resistance'] = df['high'].rolling(window=20).max().shift(1)
df['support'] = df['low'].rolling(window=20).min().shift(1)

# Determine raw breakout signals before cooldown
df['buy_base'] = (df['close'] > df['resistance']) & (df['ATR'] > df['ATR_avg']) & (df['volume'] > df['vol_avg'])
df['sell_base'] = (df['close'] < df['support']) & (df['ATR'] > df['ATR_avg']) & (df['volume'] > df['vol_avg'])

# Apply 5-candle cooldown for same signal types
cooldown = 5
last_buy_idx = -np.inf
last_sell_idx = -np.inf
df['Signal'] = 0  # 1 = Buy, -1 = Sell
for i in range(len(df)):
    if df.at[i, 'buy_base'] and (i - last_buy_idx >= cooldown):
        df.at[i, 'Signal'] = 1
        last_buy_idx = i
    elif df.at[i, 'sell_base'] and (i - last_sell_idx >= cooldown):
        df.at[i, 'Signal'] = -1
        last_sell_idx = i

# Backtest: For each signal, check next candles for TP/SL
# Example: define fixed TP=2%, SL=1% (adjust as needed)
tp_pct = 0.02
sl_pct = 0.01
lookahead = 20  # max bars to check

outcomes = []  # record (index, type, outcome)
for i in range(len(df)):
    sig = df.at[i, 'Signal']
    if sig == 0:
        continue
    entry_price = df.at[i, 'close']
    tp_price = entry_price * (1 + tp_pct) if sig == 1 else entry_price * (1 - tp_pct)
    sl_price = entry_price * (1 - sl_pct) if sig == 1 else entry_price * (1 + sl_pct)
    outcome = "Neutral"
    for j in range(i+1, min(len(df), i+lookahead+1)):
        if sig == 1:  # long position
            if df.at[j, 'high'] >= tp_price:
                outcome = "TP"
                break
            if df.at[j, 'low'] <= sl_price:
                outcome = "SL"
                break
        else:  # short position
            if df.at[j, 'low'] <= tp_price:
                outcome = "TP"
                break
            if df.at[j, 'high'] >= sl_price:
                outcome = "SL"
                break
    outcomes.append((i, sig, outcome))

# Collect signals into a DataFrame
signals = []
for idx, sig_type, outcome in outcomes:
    price = df.at[idx, 'close']
    zone = df.at[idx, 'resistance'] if sig_type == 1 else df.at[idx, 'support']
    # Distance from breakout zone (as fraction of zone price)
    if pd.notna(zone) and zone != 0:
        if sig_type == 1:
            dist = max((price - zone) / zone, 0)
        else:
            dist = max((zone - price) / zone, 0)
    else:
        dist = 0.0
    # Confidence components
    atr_flag = 1.0 if df.at[idx, 'ATR'] > df.at[idx, 'ATR_avg'] else 0.0
    vol_flag = 1.0 if df.at[idx, 'volume'] > df.at[idx, 'vol_avg'] else 0.0
    confidence = (0.4 * atr_flag) + (0.3 * vol_flag) + (0.3 * dist)
    confidence_score = round(confidence * 100, 1)  # scale to 0-100
    signals.append({
        "Time": df.at[idx, 'time'],
        "Type": "Buy" if sig_type == 1 else "Sell",
        "Entry": price,
        "ATR": round(df.at[idx, 'ATR'], 2),
        "Volume": round(df.at[idx, 'volume'], 1),
        "Distance%": f"{dist*100:.1f}%",
        "Confidence": f"{confidence_score}%",
        "Outcome": outcome
    })

if signals:
    st.subheader("Breakout Signals")
    signals_df = pd.DataFrame(signals)
    # Sort by time or keep as is
    st.table(signals_df)
else:
    st.subheader("Breakout Signals")
    st.write("No breakout signals detected in the selected range.")

