from binance.client import Client
import pandas as pd
import datetime

# Binance API credentials
api_key = 'your_api_key'
api_secret = 'your_api_secret'

client = Client(JlWm8rVqQBu7oWTBYvMZexkZWQ2uDzyXy5rEDBIvEOS8hR1vPYHt3XTZQO4KwZB4,pQsLSn50rGz3Nxur5I9wvNKR1CHkCXFevGe2Qa7hSGu0HV8lb74r3OBQO0vrsrlb)

# Define the symbol for BTC/USDT pair
symbol = 'BTCUSDT'

# Define custom start and end time
start_time = datetime.datetime(2024, 3, 15, 0, 0, 0)
end_time = datetime.datetime(2024, 6, 15, 0, 0, 0)

klines = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, start_str=str(start_time), end_str=str(end_time))

# Convert the data into a pandas dataframe for easier manipulation
df_M = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])


columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']

for col in columns_to_convert:
    df_M[col] = df_M[col].astype(float)

df_pepe_M
