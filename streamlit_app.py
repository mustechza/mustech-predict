# ✅ Install required packages (first time only):
# !pip install ipywidgets matplotlib pandas numpy TA-Lib

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipywidgets as widgets
from IPython.display import display
import talib as ta

# --- Configurable Parameters ---
sr_length = 15
sr_margin = 2
atr_period = 17
volume_sma_period = 17
TP_PCT = 0.02
SL_PCT = 0.01
DAYS_TO_PLOT = 30
tf_scale = 1  # set to 15, 60, etc., if using MTF

# --- Main Backtest Function ---
def run_backtest(df):
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    
    adj_len = int(sr_length * tf_scale)
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['vol_sma'] = df['volume'].rolling(volume_sma_period).mean()
    
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

    # --- Backtest Summary ---
    results = pd.DataFrame(trades)
    if not results.empty:
        results['duration'] = (results['exit_time'] - results['entry_time']).dt.total_seconds() / 60  # in minutes
        total_trades = len(results)
        wins = results[results['pnl'] > 0]
        losses = results[results['pnl'] < 0]
        win_rate = len(wins) / total_trades * 100
        avg_pnl = results['pnl'].mean()
        total_pnl = results['pnl'].sum()
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else np.inf

        print("\n🔍 Backtest Summary:")
        print(f"Total Trades      : {total_trades}")
        print(f"Win Rate          : {win_rate:.2f}%")
        print(f"Average PnL/trade : {avg_pnl:.4f}")
        print(f"Cumulative PnL    : {total_pnl:.4f}")
        print(f"Profit Factor     : {profit_factor:.2f}")

        results.to_csv("backtest_trades.csv", index=False)
        print("✅ Trades exported to 'backtest_trades.csv'")
    else:
        print("No trades generated.")

    # --- Plotting ---
    df_plot = df.last(f'{DAYS_TO_PLOT}D')
    entry_points = [t for t in trades if t['entry_time'] in df_plot.index and t['exit_time'] in df_plot.index]

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.6
    colors = ['green' if c >= o else 'red' for o, c in zip(df_plot['open'], df_plot['close'])]
    dates = mdates.date2num(df_plot.index.to_pydatetime())

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

    ax.set_title(f'Trade Plot (Last {DAYS_TO_PLOT} Days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Upload Widget ---
upload_btn = widgets.FileUpload(accept='.csv', multiple=False)
display(upload_btn)

def handle_upload(change):
    if upload_btn.value:
        file_data = next(iter(upload_btn.value.values()))
        content = file_data['content']
        df = pd.read_csv(io.BytesIO(content), parse_dates=['timestamp'])
        print(f"✅ File '{file_data['metadata']['name']}' uploaded. Starting backtest...")
        run_backtest(df)

upload_btn.observe(handle_upload, names='value')
