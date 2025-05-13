import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv


# Reference Documents
# https://python-binance.readthedocs.io/en/latest/index.html
# https://www.binance.com/en/binance-api
class MarketDataEndpoints:
    def __init__(self):
        load_dotenv("../api.env")
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("API key and secret must be set in environment variables.")
        self.client = Client(api_key, api_secret, testnet=False)

    def ping(self):
        return self.client.ping()

    def server_time(self):
        return self.client.get_server_time()

    def system_status(self):
        return self.client.get_system_status()

    def exchange_info(self):
        return self.client.get_exchange_info()

    def symbol_info(self, symbol):
        return self.client.get_symbol_info(symbol)

    def fetch_klines(self, symbol: str, interval: str, start_str: str | int = None, end_str: str | int = None):
        """
        "1 day ago UTC" - "1 Dec, 2017", "1 Jan, 2018" - "1 Jan, 2017"
        """
        klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
        columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
                   "Quote Asset Volume", "Number of Trades",
                   "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]
        df_klines = pd.DataFrame(klines, columns=columns)
        df_klines = df_klines.apply(pd.to_numeric, errors="raise")
        df_klines["Open Time"] = pd.to_datetime(df_klines["Open Time"], unit="ms")
        df_klines["Close Time"] = pd.to_datetime(df_klines["Close Time"], unit="ms")
        df_klines = df_klines.drop(df_klines.columns[7:], axis=1)
        df_klines = df_klines.set_index("Close Time")
        return df_klines
