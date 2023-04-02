import talib as ta
import matplotlib.pyplot as plt
import math
import numpy as np

class TechnicalAnalyst():
    def __init__(self, prices):
        self.prices = prices 

    def check_if_exceeded_stop_loss(self, price, bought_for, threshold=0.03):
        if price < bought_for*(1.00-threshold):
            return True
        else:
            return False

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

    def get_williams_oscillator(self, df, timeperiod=14):
        return ta.WILLR(df['high'], df['low'], df['close'], timeperiod=timeperiod)

    def get_average_directional_index(self, df, timeperiod=14):
        return ta.ADX(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    
    def get_directional_movement_index(self, df, timeperiod=14):
        return ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=timeperiod), ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=timeperiod)

    def get_stop_and_reverse(self, df):
        return ta.SAR(df['high'], df['low'])

    def get_moving_average_crossover(self, df, column='close', timeperiod1=5, timeperiod2=20):
        return ta.MA(df[column], timeperiod=timeperiod1), ta.MA(df[column], timeperiod=timeperiod2)

    def get_moving_average_convergence_divergence(self, df, column='close', fastperiod=12, slowperiod=26, signalperiod=9):
        return ta.MACD(df[column], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    def get_RSI_sell_buy_signals(self, data, stoploss=False, threshold=0.03):
        bought_for = 0
        area = []
        signals = []

        for idx, (i, j) in enumerate(zip(data, self.prices)):
            area.append('down' if i < 30 else 'top' if i > 70 else 'middle')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'middle' and last != 1:
                signals.append(1)
                bought_for = j
            elif area[idx-1] == 'top' and area[idx] == 'middle' and last != 2:
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(j, bought_for, threshold=threshold) and last != 2 and stoploss:
                    signals.append(2)
                else:
                    signals.append(0)

        return signals
    
    def get_ROC_sell_buy_signals(self, data, stoploss=False, threshold=0.03):
        bought_for = 0
        area = []
        signals = []
        for idx, (i, j) in enumerate(zip(data, self.prices)):
            area.append('down' if i < 0 else 'top')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'top' and last != 1:
                signals.append(1)
                bought_for = j
            elif area[idx-1] == 'top' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(j, bought_for, threshold=threshold) and last != 2 and stoploss:
                    signals.append(2)
                else:
                    signals.append(0)

        return signals
    
    def get_MAC_sell_buy_signals(self, ma1, ma2, stoploss=False, threshold=0.03):
        bought_for = 0
        area = []
        signals = []
        for idx, (i, j, p) in enumerate(zip(ma1, ma2, self.prices)):
            area.append('down' if i < j else 'top')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'top' and last != 1:
                signals.append(1)
                bought_for = p
            elif area[idx-1] == 'top' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(p, bought_for, threshold=threshold) and last != 2 and stoploss:
                    signals.append(2)
                else:
                    signals.append(0)

        return signals
    
    def get_CCI_sell_buy_signals(self, data, stoploss=False, threshold=0.03):
        bought_for = 0
        area = []
        signals = []

        for idx, (i, j) in enumerate(zip(data, self.prices)):
            area.append('down' if i < -100 else 'top' if i > 100 else 'middle')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'middle' and area[idx] == 'top' and last != 1:
                signals.append(1)
                bought_for = j
            elif area[idx-1] == 'middle' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(j, bought_for, threshold=threshold) and last != 2 and stoploss:
                    signals.append(2)
                else:
                    signals.append(0)
        return signals
    
    def get_WO_sell_buy_signals(self, data, stoploss=False, threshold=0.03):
        bought_for = 0
        area = []
        signals = []
        for idx, (i, j) in enumerate(zip(data, self.prices)):
            area.append('down' if i < -80 else 'top' if i > -20 else 'middle')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'middle' and last != 1:
                signals.append(1)
                bought_for = j
            elif area[idx-1] == 'top' and area[idx] == 'middle' and last != 2:
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(j, bought_for, threshold=threshold) and last != 2 and stoploss:
                    signals.append(2)
                else:
                    signals.append(0)
        return signals
    
    def get_SO_sell_buy_signals(self, k, d, stoploss=False, threshold=0.03):
        bought_for = 0
        area = []
        signals = []
        for idx, (i, j, p) in enumerate(zip(k, d, self.prices)):
            area.append('down' if i <= j else 'top')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if i < 20 and j < 20 and area[idx-1] == 'down' and area[idx] == 'top' and last != 1:
                signals.append(1)
                bought_for = p
            elif i > 80 and j > 80 and area[idx-1] == 'top' and area[idx] == 'down' and last != 2:
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(p, bought_for, threshold=threshold) and last != 2 and stoploss:
                    signals.append(2)
                else:
                    signals.append(0)

        return signals

    def get_SAR_sell_buy_signals(self, data, stoploss=False, threshold=0.03):
        adx = self.get_average_directional_index(data)
        plus_di, minus_di = self.get_directional_movement_index(data)
        sar = self.get_stop_and_reverse(data)

        bought_for = 0
        data = data['close'].tolist()
        area = []
        signals = []
        for idx, (i, j, p, m, a, pr) in enumerate(zip(data, sar, plus_di, minus_di, adx, self.prices)):
            area.append('down' if i <= j else 'top')
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)

            if area[idx-1] == 'down' and area[idx] == 'top' and last != 1 and a > 40:   # a to też dodatkowy feature
                signals.append(1)
                bought_for = pr
            elif area[idx-1] == 'top' and area[idx] == 'down' and last != 2 and p < m and a > 40:  # to jest dodatkowy feature and p < m
                signals.append(2)
            else:
                if self.check_if_exceeded_stop_loss(pr, bought_for, threshold=threshold) and last != 2 and stoploss:
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

        number_of_transactions = sum(1 if i != 0 else 0 for i in signals)
        score = np.round(((cash/initial_cash)-1)*100, 2)

        return name, np.round(cash,2), np.round(score,2), number_of_transactions, np.round((cash-initial_cash)/number_of_transactions,2) if number_of_transactions > 0 else 0

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