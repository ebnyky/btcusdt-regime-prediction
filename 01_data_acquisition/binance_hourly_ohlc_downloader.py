# python script to download btcusdt data

import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests


# Binance public market-data endpoint
URL = "https://data-api.binance.vision/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 1000

# Beginning of 7 July 2025, UTC
START_DATE = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# End date is exclusive, so this includes all candles on 31 July 2026
END_DATE = datetime(2025, 7, 7, 0, 0, 0, tzinfo=timezone.utc)

OUTPUT_FILE = Path.home()/"OneDrive/Desktop/BTCUSDT_1h_2020-01-01_to_2025-07-06.csv"


def datetime_to_milliseconds(date_value):
    """Convert a timezone-aware datetime to Unix milliseconds."""
    return int(date_value.timestamp() * 1000)


def download_ohlc():
    start_ms = datetime_to_milliseconds(START_DATE)
    end_ms = datetime_to_milliseconds(END_DATE)

    current_start = start_ms
    all_candles = []

    while current_start < end_ms:
        parameters = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": current_start,
            # Binance treats endTime as inclusive, so subtract 1 millisecond.
            "endTime": end_ms - 1,
            "limit": LIMIT,
        }

        try:
            response = requests.get(
                URL,
                params=parameters,
                timeout=30,
            )
            response.raise_for_status()

        except requests.RequestException as error:
            print(f"Request failed: {error}")
            print("Trying again in 5 seconds...")
            time.sleep(5)
            continue

        candles = response.json()

        if not candles:
            break

        all_candles.extend(candles)

        # Open time of the final candle returned
        last_open_time = candles[-1][0]

        # Move forward by one hour to avoid downloading it again
        current_start = last_open_time + (60 * 60 * 1000)

        print(
            f"Downloaded {len(all_candles)} candles. "
            f"Latest candle: "
            f"{datetime.fromtimestamp(last_open_time / 1000, tz=timezone.utc)}"
        )

        # Small pause to avoid sending requests too quickly
        time.sleep(0.2)

    return all_candles


def save_ohlc(candles):
    if not candles:
        print("No candles were downloaded.")
        return

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    dataframe = pd.DataFrame(candles, columns=columns)

    # Keep only the requested OHLC data and timestamp
    dataframe = dataframe[
        ["open_time", "open", "high", "low", "close"]
    ].copy()

    # Convert timestamps to readable UTC dates
    dataframe["open_time"] = pd.to_datetime(
        dataframe["open_time"],
        unit="ms",
        utc=True,
    )

    # Convert price columns from strings to numbers
    price_columns = ["open", "high", "low", "close"]

    dataframe[price_columns] = dataframe[price_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove duplicates, just in case
    dataframe = dataframe.drop_duplicates(
        subset="open_time",
        keep="first",
    )

    dataframe = dataframe.sort_values("open_time").reset_index(drop=True)

    dataframe.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload complete.")
    print(f"Number of candles: {len(dataframe)}")
    print(f"First candle: {dataframe['open_time'].iloc[0]}")
    print(f"Last candle:  {dataframe['open_time'].iloc[-1]}")
    print(f"Saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    downloaded_candles = download_ohlc()
    save_ohlc(downloaded_candles)
