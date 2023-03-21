import pandas as pd
import datetime
from binance.exceptions import BinanceAPIException
import logging

class API():
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='info.txt',
                    filemode='a',
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    
    def get_all_tickers(self):
        try:
            tickers = self.client.get_all_tickers()
            return pd.DataFrame(tickers).set_index('symbol')
        except:
            self.logger.info("fetching all tickers failed")
            return None
    
    def get_price(self, symbol):
        try:
            return float(self.get_all_tickers().loc[symbol]['price'])
        except:
            self.logger.info("fetching price failed")
            return None
    
    def get_order_book(self, symbol, type):
        assert type in ['bids','asks']
        depth = self.client.get_order_book(symbol=symbol)
        return pd.DataFrame(depth[type], columns=['price', 'volume'])

    def get_historical_klines(self, symbol, interval, days=7):
        start = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        start = start.strftime("%Y-%m-%d %H:%M:%S")
        historical = self.client.get_historical_klines(symbol, interval, start, end_str=None)
        df = pd.DataFrame(historical, columns=['open time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote asset volume', 'number of trades', 'tb base volume', 'tb quote volume', 'ignore'])
        df['open time'] = pd.to_datetime(df['open time']/1000, unit='s')
        df['close time'] = pd.to_datetime(df['close time']/1000, unit='s')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote asset volume', 'tb base volume', 'tb quote volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
        return df
    
    def get_highest_price(self, klines):
        return max(list(map(float, klines['high'].tolist())))
    
    def current_to_highest_price_ratio(self, symbol, klines):
        highest = self.get_highest_price(klines)
        current = self.get_price(symbol)
        return round(((current/highest)-1)*100, 2)
    
    def get_crypto_with_nonzero_balance(self):
        return [i['asset'] for i in self.client.get_account()['balances'] if float(i['free']) > 0]
        


    def get_favorites(self):
        pass
        # try:
        #     account_info = self.client.get_account()
        #     # watchlist_symbols = [asset['asset'] for asset in account_info['userData']['watchList']]
        #     print(account_info)
        #     exit()
        #     favorites = account_info['userAssets']
        #     return [asset['asset'] for asset in favorites if asset['favorite']]
        # except BinanceAPIException as e:
        #     print(e)