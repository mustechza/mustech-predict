import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Sidebar Parameters ---
st.sidebar.title("Settings")
sr_length = st.sidebar.slider("Pivot Lookback Length", 5, 30, 15)
sr_margin = st.sidebar.slider("Zone Margin", 1, 5, 2)
atr_period = st.sidebar.slider("ATR Period", 10, 30, 17)
volume_sma_period = st.sidebar.slider("Volume SMA Period", 10, 30, 17)
TP_PCT = st.sidebar.slider("Take Profit %", 0.5, 10.0, 2.0) / 100
SL_PCT = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 1.0) / 100
DAYS_TO_PLOT = st.sidebar.slider("Days to Plot", 1, 90, 30)
tf_scale = st.sidebar.slider("Timeframe Scale", 1, 30, 1)

# --- Load Data ---
uploaded_file = st.file_uploader("Upload your OHLCV CSV", type="csv")
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# --- Technicals ---
adj_len = int(sr_length * tf_scale)
df['ATR'] = df.ta.atr(length=atr_period)
df['vol_sma'] = df['volume'].rolling(volume_sma_period).mean()

# --- Pivot Zones ---
df['pivothigh'] = df['high'][(df['high'].shift(adj_len) < df['high']) & (df['high'].shift(-adj_len) < df['high'])]
df['pivotlow'] = df['low'][(df['low'].shift(adj_len) < df['low']) & (df['low'].shift(-adj_len) < df['low'])]

zone_range = (df['high'].max() - df['low'].min()) / df['high'].max()
resistance_zones = []
support_zones = []

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

# --- Breakout Logic ---
df['bull_breakout'] = (df['close'] > df['resistance'].shift(1)) & (df['close'].shift(1) <= df['resistance'].shift(1))
df['bear_breakout'] = (df['close'] < df['support'].shift(1)) & (df['close'].shift(1) >= df['support'].shift(1))
df['vol_tag'] = np.where(df['volume'] > 4.669 * df['vol_sma'], 'V-Spike',
                  np.where(df['volume'] > 1.618 * df['vol_sma'], 'High',
                  np.where(df['volume'] < 0.618 * df['vol_sma'], 'Low', 'Avg')))

# --- Trade Tracking ---
trades = []
position = None

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    if position:
        current_price = row['close']
        entry_price = position['entry_price']
        if position['type'] == 'long':
            if current_price >= entry_price * (1 + TP_PCT):
                trades.append({'type': 'long', 'entry': position['entry_time'], 'exit': row.name, 'pnl': TP_PCT})
                position = None
            elif current_price <= entry_price * (1 - SL_PCT):
                trades.append({'type': 'long', 'entry': position['entry_time'], 'exit': row.name, 'pnl': -SL_PCT})
                position = None
        elif position['type'] == 'short':
            if current_price <= entry_price * (1 - TP_PCT):
                trades.append({'type': 'short', 'entry': position['entry_time'], 'exit': row.name, 'pnl': TP_PCT})
                position = None
            elif current_price >= entry_price * (1 + SL_PCT):
                trades.append({'type': 'short', 'entry': position['entry_time'], 'exit': row.name, 'pnl': -SL_PCT})
                position = None

    if position is None:
        if row['bull_breakout']:
            position = {'type': 'long', 'entry_price': row['close'], 'entry_time': row.name}
        elif row['bear_breakout']:
            position = {'type': 'short', 'entry_price': row['close'], 'entry_time': row.name}

# --- Plotting ---
df_plot = df.last(f'{DAYS_TO_PLOT}D').copy()
entry_points = [t for t in trades if t['entry'] in df_plot.index and t['exit'] in df_plot.index]

fig, ax = plt.subplots(figsize=(14, 6))
width = 0.6
colors = ['green' if c >= o else 'red' for o, c in zip(df_plot['open'], df_plot['close'])]
dates = mdates.date2num(df_plot.index.to_pydatetime())

for i in range(len(df_plot)):
    ax.plot([dates[i], dates[i]], [df_plot['low'].iloc[i], df_plot['high'].iloc[i]], color='black')
    ax.bar(dates[i], df_plot['close'].iloc[i] - df_plot['open'].iloc[i], bottom=df_plot['open'].iloc[i],
           width=width, color=colors[i], edgecolor='black')

ax.plot(dates, df_plot['resistance'], label='Resistance', color='red', linestyle='--')
ax.plot(dates, df_plot['support'], label='Support', color='green', linestyle='--')

for t in entry_points:
    entry_idx = df_plot.index.get_loc(t['entry'])
    exit_idx = df_plot.index.get_loc(t['exit'])
    entry_price = df_plot.loc[t['entry'], 'close']
    exit_price = df_plot.loc[t['exit'], 'close']
    
    if t['type'] == 'long':
        ax.scatter(dates[entry_idx], entry_price, color='blue', marker='^', label='Long Entry')
        ax.scatter(dates[exit_idx], exit_price, color='blue', marker='v', label='Long Exit')
    else:
        ax.scatter(dates[entry_idx], entry_price, color='orange', marker='v', label='Short Entry')
        ax.scatter(dates[exit_idx], exit_price, color='orange', marker='^', label='Short Exit')

ax.set_title(f"Breakout Signal Trades (Last {DAYS_TO_PLOT} Days)")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Backtest Stats ---
results = pd.DataFrame(trades)
st.subheader("Backtest Summary")
if not results.empty:
    total_trades = len(results)
    wins = results[results['pnl'] > 0]
    losses = results[results['pnl'] < 0]
    win_rate = len(wins) / total_trades * 100
    avg_pnl = results['pnl'].mean()
    total_pnl = results['pnl'].sum()
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else np.inf

    st.markdown(f"""
    - **Total Trades**: {total_trades}
    - **Win Rate**: {win_rate:.2f}%
    - **Average PnL/trade**: {avg_pnl:.4f}
    - **Cumulative PnL**: {total_pnl:.4f}
    - **Profit Factor**: {profit_factor:.2f}
    """)
else:
    st.info("No trades were generated based on the current strategy.")
