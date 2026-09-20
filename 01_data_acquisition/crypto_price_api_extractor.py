### extracting historical data of sonic/usdt pair
##
##import ccxt
##
### Choose exchange
##exchange = ccxt.binance()   # or ccxt.binance(), ccxt.bitget(), etc.
##
####symbol = 'S/USDT'     # trading pair format for ccxt
####
##### Fetch current ticker
####ticker = exchange.fetch_ticker(symbol)
####print("Last price:", ticker['last'])
####print("High/Low:", ticker['high'], ticker['low'])
####print("Open/Close", ticker["open"], ticker['close'])
####print("Volume:", ticker['baseVolume'])
####
####
####
####import time
####
##### Use Bybit or another exchange
####exchange = ccxt.binance()
####
####symbol = 'S/USDT'
####timeframe = '1h'   # change to '1m', '1d', etc.
####
##### Since many exchanges limit history per call, you may loop older data
####holder = []
####ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
####for candle in ohlcv:
####    print(candle)
####    if len(holder) < 10:
####        holder.append(candle)
##
##exchange.load_markets()
##print("S/USDT" in exchange.symbols)
##
##
##
##import ccxt
##import pandas as pd
##from datetime import datetime
##
### Initialize Binance
##import ccxt
##import pandas as pd
##
##exchange = ccxt.binance({'enableRateLimit': True})
##
##symbol = 'S/USDT'
##timeframe = '1m'
##limit = 1000
##
##since = exchange.parse8601("2017-01-01T00:00:00Z")  # safely early
##all_candles = []
##
##try:
##    while True:
##        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
##
##        if not candles:
##            break
##
##        # Prevent duplicate last candle
##        if all_candles and candles[0][0] <= all_candles[-1][0]:
##            candles = candles[1:]
##
##        all_candles.extend(candles)
##
##        since = candles[-1][0] + 1
##
##        if len(candles) < limit:
##            break
##except PermissionError:
##    print("Extraction Done.")
##    print()
##
##print(f"Total candles fetched: {len(all_candles)}")
##
##df = pd.DataFrame(
##    all_candles,
##    columns=["timestamp", "open", "high", "low", "close", "volume"]
##)
##df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
##
##
##df.to_csv(r"C:\Users\ebnyk\OneDrive\Desktop\SONIC_USDT_binance_ohlcv_minute.csv", index=False)
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
