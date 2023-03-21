from dotenv import load_dotenv
import os
import pandas as pd
from interface.api import API
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

from interface.technical_analysis import TechnicalAnalyst

import os
import sys
import subprocess
from interface.analyzing import Analizer
import matplotlib.pyplot as plt

load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


def run_in_background():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    exe = os.path.abspath(sys.executable)
    args = [os.path.abspath(__file__)]
    subprocess.Popen([exe] + args, startupinfo=si)

if __name__ == '__main__':
    # run_in_background()    
    client = Client(API_KEY, SECRET_KEY)

    api = API(client)
    analizer = Analizer(api)
    tm = TechnicalAnalyst()

    data = api.get_historical_klines("BTCBUSD", client.KLINE_INTERVAL_1DAY, days=3000)
    print(f'days: {len(data)}')
    
    rsi = tm.get_relative_strength_index(data)
    rsi_signals = tm.get_RSI_sell_buy_signals(rsi)

    roc = tm.get_rate_of_change(data)
    roc_signals = tm.get_ROC_sell_buy_signals(roc)

    cci = tm.get_commodity_channel_index(data)
    cci_signals = tm.get_CCI_sell_buy_signals(cci)

    k, d = tm.get_stochastic_oscillator(data)
    so_signals = tm.get_SO_sell_buy_signals(k, d)

    wo = tm.get_williams_oscillator(data)
    wo_signals = tm.get_WO_sell_buy_signals(wo)

    adx = tm.get_average_directional_index(data)
    plus_di, minus_di = tm.get_directional_movement_index(data)
    sar = tm.get_stop_and_reverse(data)

    sar_signals = tm.get_SAR_sell_buy_signals(data, sar, plus_di, minus_di, adx)

    # plt.plot(adx)
    # plt.axhline(y=25)
    # plt.show()
    # plt.clf()

    # plt.plot(k, c='blue')
    # plt.plot(d, c='red')
    # plt.axhline(y=80)
    # plt.axhline(y=20)
    # plt.show()
    # plt.clf()
    
    tm.calculate_profit(data, rsi_signals, name='RSI')
    tm.calculate_profit(data, roc_signals, name='ROC')
    tm.calculate_profit(data, cci_signals, name='CCI')
    tm.calculate_profit(data, so_signals, name='SO')
    tm.calculate_profit(data, wo_signals, name='WO')
    tm.calculate_profit(data, sar_signals, name='SAR')
    # tm.draw_price_chart(data, sar_signals)

    exit()
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
