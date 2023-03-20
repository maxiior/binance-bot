from dotenv import load_dotenv
import os
import pandas as pd
from interface.api import API
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

import os
import sys
import subprocess
import mplfinance as mpf
from interface.analyzing import Analizer


def running_in_background():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    exe = os.path.abspath(sys.executable)
    args = [os.path.abspath(__file__)]
    subprocess.Popen([exe] + args, startupinfo=si)

if __name__ == '__main__':
    # running_in_background()
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    client = Client(API_KEY, SECRET_KEY)

    api = API(client)
    analizer = Analizer(api)

    # r.get_price('BTCUSDT')
    # r.get_order_book('GNSUSDT', type='bids')

    klines = api.get_historical_klines('HFTBUSD', client.KLINE_INTERVAL_1HOUR, days=7)
    ratio = api.current_to_highest_price_ratio('HFTBUSD', klines)
    print(ratio)

    # print(api.get_favorites())

    x = api.get_crypto_with_nonzero_balance()
    # analizer.cyclical_analysis(x)

    close = klines['high'].tolist()


    # analizer.get_extremes(close)
    # close = analizer.get_continuous_avg(close_original, size=24)
    extremes = analizer.get_extremes(close)
    extremes = analizer.extremes_convolution(extremes, close, size=12, type='min')

    values = [close[i] for i in extremes]

    print(values)

    import matplotlib.pyplot as plt

    plt.plot(close)
    for x, y in zip(extremes,values):
        plt.scatter(x, y, color='red')
    plt.show()


    # mpf.plot(klines.set_index('close time'))