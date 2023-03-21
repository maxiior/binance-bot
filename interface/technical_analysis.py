import talib as ta
import matplotlib.pyplot as plt
import math
import numpy as np

class TechnicalAnalyst():
    def __init__(self) -> None:
        pass

    def get_simple_moving_average(self, df, column='close', time_period=100):
        return ta.SMA(df[column], time_period).tolist()
    
    def get_relative_strength_index(self, df, column='close'):
        return ta.RSI(df[column]).tolist()

    def get_rate_of_change(self, df, column='close', timeperiod=14):
        return ta.ROC(df[column], timeperiod=timeperiod).tolist()
    
    def get_commodity_channel_index(self, df, timeperiod=14):
        return ta.CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod).tolist()

    def get_stochastic_oscillator(self, df, k=5, d=15):
        return ta.STOCH(df['high'], df['low'], df['close'], fastk_period=k, slowk_period=d, slowk_matype=ta.MA_Type.SMA, slowd_period=d, slowd_matype=ta.MA_Type.SMA)

    def get_RSI_sell_buy_signals(self, data):
        area = []
        signals = []

        for idx, i in enumerate(data):
            area.append('down' if i < 30 else 'top' if i > 70 else 'middle')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'middle' and last != 1:
                signals.append(1)
            elif area[idx-1] == 'top' and area[idx] == 'middle' and last != 2:
                signals.append(2)
            else:
                signals.append(0)

        return signals
    
    def get_ROC_sell_buy_signals(self, data):
        area = []
        signals = []
        for idx, i in enumerate(data):
            area.append('down' if i < 0 else 'top')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'top' and last != 1:
                signals.append(1)
            elif area[idx-1] == 'top' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                signals.append(0)

        return signals
    
    def get_CCI_sell_buy_signals(self, data):
        area = []
        signals = []
        for idx, i in enumerate(data):
            area.append('down' if i < -100 else 'top' if i > 100 else 'middle')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'middle' and area[idx] == 'top' and last != 1:
                signals.append(1)
            elif area[idx-1] == 'middle' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                signals.append(0)
        return signals
    
    def get_SO_sell_buy_signals(self, k, d):
        area = []
        signals = []
        for idx, (i, j) in enumerate(zip(k, d)):
            area.append('down' if i <= j else 'top')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if i < 30 and j < 30:
                print(i,j)

            if i < 30 and j < 30 and area[idx-1] == 'down' and area[idx] == 'top' and last != 1:
                signals.append(1)
            elif i > 70 and j > 70 and area[idx-1] == 'top' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                signals.append(0)

        return signals

    def calculate_profit(self, df, signals, name, column='close', cash=1000000.0):
        initial_cash, cash_before = cash, cash
        data = df[column].tolist()
        amount = 0
        for i, j in zip(signals, data):
            if i == 1:
                amount = math.floor(cash/j)
                cash -= amount * j
            elif i == 2:
                cash += amount * j
                # print(f'{np.round(((cash/cash_before)-1)*100, 2)}%')
                cash_before = cash
                amount = 0
        
        if amount != 0:
            cash += amount * j
            # print(f'{np.round(((cash/cash_before)-1)*100, 2)}%')
            cash_before = cash
            amount = 0

        print(f'final result ({name}): {np.round(cash,2)}$ ({np.round(((cash/initial_cash)-1)*100, 2)}%)')

    def draw_price_chart(self, df, signals, column='close'):
        data = df[column].tolist()
        x1 = [i for i, val in enumerate(signals) if val == 1]
        x2 = [i for i, val in enumerate(signals) if val == 2]
                
        values1 = [data[i] for i in x1]
        values2 = [data[i] for i in x2]

        plt.plot(data)
        plt.scatter(x1, values1, c='green')
        plt.scatter(x2, values2, c='red')
        plt.show()
        plt.clf()