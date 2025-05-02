import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib as ta

# --- Settings ---
sr_length = 15
sr_margin = 2
atr_period = 17
volume_sma_period = 17
TP_PCT = 0.02
SL_PCT = 0.01
DAYS_TO_PLOT = 30
tf_scale = 1  # Use 15, 60, etc., for MTF logic

st.set_page_config(layout="wide")
st.title("📉 LuxAlgo-Style Support/Resistance Backtest")

uploaded_file = st.file_uploader("Upload your OHLCV CSV", type="csv")

def run_backtest(df):
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    adj_len = int(sr_length * tf_scale)
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['vol_sma'] = df['volume'].rolling(volume_sma_period).mean()

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

    df['bull_breakout'] = (df['close'] > df['resistance'].shift(1)) & (df['close'].shift(1) <= df['resistance'].shift(1))
    df['bear_breakout'] = (df['close'] < df['support'].shift(1)) & (df['close'].shift(1) >= df['support'].shift(1))

    # --- Trade Tracking ---
    trades = []
    position = None
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position:
            price = row['close']
            entry = position['entry_price']
            if position['type'] == 'long':
                if price >= entry * (1 + TP_PCT):
                    trades.append({**position, 'exit_time': row.name, 'exit_price': price, 'pnl': TP_PCT})
                    position = None
                elif price <= entry * (1 - SL_PCT):
                    trades.append({**position, 'exit_time': row.name, 'exit_price': price, 'pnl': -SL_PCT})
                    position = None
            else:
                if price <= entry * (1 - TP_PCT):
                    trades.append({**position, 'exit_time': row.name, 'exit_price': price, 'pnl': TP_PCT})
                    position = None
                elif price >= entry * (1 + SL_PCT):
                    trades.append({**position, 'exit_time': row.name, 'exit_price': price, 'pnl': -SL_PCT})
                    position = None
        if position is None:
            if row['bull_breakout']:
                position = {'type': 'long', 'entry_price': row['close'], 'entry_time': row.name}
            elif row['bear_breakout']:
                position = {'type': 'short', 'entry_price': row['close'], 'entry_time': row.name}

    # --- Plotting ---
    st.subheader("📊 Price Chart with Trades")
    df_plot = df.last(f'{DAYS_TO_PLOT}D')
    entry_points = [t for t in trades if t['entry_time'] in df_plot.index and t['exit_time'] in df_plot.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    dates = mdates.date2num(df_plot.index.to_pydatetime())
    width = 0.6
    colors = ['green' if c >= o else 'red' for o, c in zip(df_plot['open'], df_plot['close'])]

    for i in range(len(df_plot)):
        ax.plot([dates[i], dates[i]], [df_plot['low'].iloc[i], df_plot['high'].iloc[i]], color='black')
        ax.bar(dates[i], df_plot['close'].iloc[i] - df_plot['open'].iloc[i], bottom=df_plot['open'].iloc[i],
               width=width, color=colors[i], edgecolor='black')

    ax.plot(dates, df_plot['resistance'], label='Resistance', color='red', linestyle='--', alpha=0.6)
    ax.plot(dates, df_plot['support'], label='Support', color='green', linestyle='--', alpha=0.6)

    for t in entry_points:
        entry_idx = df_plot.index.get_loc(t['entry_time'])
        exit_idx = df_plot.index.get_loc(t['exit_time'])
        ax.scatter(dates[entry_idx], t['entry_price'], color='blue' if t['type'] == 'long' else 'orange', marker='^')
        ax.scatter(dates[exit_idx], t['exit_price'], color='blue' if t['type'] == 'long' else 'orange', marker='v')

    ax.set_title(f'Trades (Last {DAYS_TO_PLOT} Days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Summary Stats ---
    results = pd.DataFrame(trades)
    st.subheader("📈 Backtest Summary")

    if not results.empty:
        results['duration_min'] = (results['exit_time'] - results['entry_time']).dt.total_seconds() / 60
        total_trades = len(results)
        wins = results[results['pnl'] > 0]
        losses = results[results['pnl'] < 0]
        win_rate = len(wins) / total_trades * 100
        avg_pnl = results['pnl'].mean()
        total_pnl = results['pnl'].sum()
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else np.inf

        st.metric("Total Trades", total_trades)
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.metric("Avg PnL per Trade", f"{avg_pnl:.4f}")
        st.metric("Cumulative PnL", f"{total_pnl:.4f}")
        st.metric("Profit Factor", f"{profit_factor:.2f}")

        st.download_button("📥 Download Trades CSV", data=results.to_csv(index=False), file_name="backtest_trades.csv")
    else:
        st.warning("No trades were generated.")

# --- Upload trigger ---
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    st.success("✅ File loaded. Running backtest...")
    run_backtest(df)
