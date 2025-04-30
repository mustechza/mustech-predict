import streamlit as st
import pandas as pd
import numpy as np
import talib as ta
import plotly.graph_objs as go
from datetime import timedelta

# --- Sidebar Inputs ---
st.sidebar.title("Breakout Strategy Settings")
sr_length = st.sidebar.slider("Detection Length", 5, 50, 15)
sr_margin = st.sidebar.slider("Support/Resistance Margin", 0.5, 5.0, 2.0, step=0.1)
tp_pct = st.sidebar.slider("Take Profit (%)", 0.5, 5.0, 2.0) / 100
sl_pct = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.0) / 100
days_to_plot = st.sidebar.slider("Days to Plot", 5, 60, 30)

# --- Upload CSV ---
st.title("📈 Breakout Signal Dashboard")
uploaded_file = st.file_uploader("Upload OHLCV CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # --- Indicators ---
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=17)
    df['vol_sma'] = df['volume'].rolling(17).mean()

    # --- Pivot Points ---
    adj_len = sr_length
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

    df['resistance'], df['support'] = np.nan, np.nan
    for zone in resistance_zones:
        df.loc[df.index >= zone['index'], 'resistance'] = zone['top']
    for zone in support_zones:
        df.loc[df.index >= zone['index'], 'support'] = zone['bottom']

    # --- Breakout Logic ---
    df['bull_breakout'] = (df['close'] > df['resistance'].shift(1)) & (df['close'].shift(1) <= df['resistance'].shift(1))
    df['bear_breakout'] = (df['close'] < df['support'].shift(1)) & (df['close'].shift(1) >= df['support'].shift(1))

    # --- Trades ---
    trades = []
    position = None

    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i-1]
        if position:
            entry_price = position['entry_price']
            if position['type'] == 'long':
                if row['close'] >= entry_price * (1 + tp_pct):
                    trades.append({'type': 'long', 'entry': position['entry_time'], 'exit': row.name, 'pnl': tp_pct})
                    position = None
                elif row['close'] <= entry_price * (1 - sl_pct):
                    trades.append({'type': 'long', 'entry': position['entry_time'], 'exit': row.name, 'pnl': -sl_pct})
                    position = None
            elif position['type'] == 'short':
                if row['close'] <= entry_price * (1 - tp_pct):
                    trades.append({'type': 'short', 'entry': position['entry_time'], 'exit': row.name, 'pnl': tp_pct})
                    position = None
                elif row['close'] >= entry_price * (1 + sl_pct):
                    trades.append({'type': 'short', 'entry': position['entry_time'], 'exit': row.name, 'pnl': -sl_pct})
                    position = None
        if position is None:
            if row['bull_breakout']:
                position = {'type': 'long', 'entry_price': row['close'], 'entry_time': row.name}
            elif row['bear_breakout']:
                position = {'type': 'short', 'entry_price': row['close'], 'entry_time': row.name}

    # --- Chart Plot ---
    df_plot = df.last(f"{days_to_plot}D").copy()
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'],
        name='Candles'))

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['resistance'],
        mode='lines', name='Resistance', line=dict(color='red', dash='dot')))

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['support'],
        mode='lines', name='Support', line=dict(color='green', dash='dot')))

    for t in trades:
        if t['entry'] in df_plot.index and t['exit'] in df_plot.index:
            color = 'blue' if t['type'] == 'long' else 'orange'
            entry_price = df_plot.loc[t['entry'], 'close']
            exit_price = df_plot.loc[t['exit'], 'close']
            fig.add_trace(go.Scatter(
                x=[t['entry']], y=[entry_price],
                mode='markers', marker=dict(color=color, symbol='triangle-up'), name=f'{t["type"].capitalize()} Entry'))
            fig.add_trace(go.Scatter(
                x=[t['exit']], y=[exit_price],
                mode='markers', marker=dict(color=color, symbol='triangle-down'), name=f'{t["type"].capitalize()} Exit'))

    fig.update_layout(title="Support/Resistance Breakout Strategy", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # --- Trade Summary ---
    st.subheader("📊 Backtest Summary")
    results = pd.DataFrame(trades)
    if not results.empty:
        total = len(results)
        wins = results[results['pnl'] > 0]
        losses = results[results['pnl'] < 0]
        win_rate = len(wins) / total * 100
        total_pnl = results['pnl'].sum()
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else np.inf

        st.metric("Total Trades", total)
        st.metric("Win Rate (%)", f"{win_rate:.2f}")
        st.metric("Total PnL (%)", f"{total_pnl*100:.2f}")
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        st.dataframe(results)
    else:
        st.warning("No trades generated using current settings.")

else:
    st.info("Please upload a CSV file with columns: timestamp, open, high, low, close, volume")
