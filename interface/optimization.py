import pandas as pd
from technical_analysis import TechnicalAnalyst


def get_optimalized_thresholds(tm, values, data, indicator='RSI', stoploss=False, stoploss_threshold=0.03):
    final_profit = 0

    if indicator in ['RSI']:
        final_lower_limit, final_upper_limit = 30, 70

        for i in range(1,101):
            for j in range(1, 101):
                signals = tm.get_RSI_sell_buy_signals(values, stoploss=stoploss, threshold=stoploss_threshold, lower_limit=i, upper_limit=j)
                profit = tm.calculate_profit(data, signals, name=indicator, cash=1000000)[2]
                
                if profit > final_profit:
                    final_profit = profit
                    final_lower_limit = i
                    final_upper_limit = j
        
        return final_lower_limit, final_upper_limit
    elif indicator in ['ROC']:
        final_limit = 0

        for i in range(-100,101):
            signals = tm.get_ROC_sell_buy_signals(values, stoploss=stoploss, threshold=stoploss_threshold, limit=i)
            profit = tm.calculate_profit(data, signals, name=indicator, cash=1000000)[2]
            if profit > final_profit:
                final_profit = profit
                final_limit = i
        
        return final_limit
    elif indicator in ['CCI']:
        final_lower_limit, final_upper_limit = -100, 100

        for i in range(-100,101):
            for j in range(-100,101):
                signals = tm.get_CCI_sell_buy_signals(values, stoploss=stoploss, threshold=stoploss_threshold, lower_limit=i, upper_limit=j)
                profit = tm.calculate_profit(data, signals, name=indicator, cash=1000000)[2]
                
                if profit > final_profit:
                    final_profit = profit
                    final_lower_limit = i
                    final_upper_limit = j
    elif indicator in ['SO']:
        final_lower_limit, final_upper_limit = 20, 80

        for i in range(1,101):
            for j in range(1,101):
                signals = tm.get_SO_sell_buy_signals(values[0], values[1], stoploss=stoploss, threshold=stoploss_threshold, lower_limit=i, upper_limit=j)
                profit = tm.calculate_profit(data, signals, name=indicator, cash=1000000)[2]
                
                if profit > final_profit:
                    final_profit = profit
                    final_lower_limit = i
                    final_upper_limit = j

        return final_lower_limit, final_upper_limit
    elif indicator in ['WO']:
        final_lower_limit, final_upper_limit = -20, -80

        for i in range(-100,1):
            for j in range(-100,1):
                signals = tm.get_WO_sell_buy_signals(values, stoploss=stoploss, threshold=stoploss_threshold, lower_limit=i, upper_limit=j)
                profit = tm.calculate_profit(data, signals, name=indicator, cash=1000000)[2]
                
                if profit > final_profit:
                    final_profit = profit
                    final_lower_limit = i
                    final_upper_limit = j
            
        return final_lower_limit, final_upper_limit
    elif indicator in ['MAC']:
        final_m1, final_m2 = 5, 20

        for i in range(1,101):
            for j in range(1,101):
                m1, m2 = tm.get_moving_average_crossover(data, timeperiod1=i, timeperiod2=j)
                signals = tm.get_MAC_sell_buy_signals(m1, m2, stoploss=stoploss, threshold=stoploss_threshold)
                profit = tm.calculate_profit(data, signals, name=indicator, cash=1000000)[2]
                
                if profit > final_profit:
                    final_profit = profit
                    final_m1 = i
                    final_m2 = j
        print(final_m1, final_m2)
        return final_m1, final_m2


def get_oplimalized_signals(data, indicator='RSI', stoploss=False, stoploss_threshold=0.03):
    signals = []
    area = []
    bought_for = 0

    if indicator == 'MAC':
        tm = TechnicalAnalyst(prices=data[0]['close'].tolist())
        m1, m2 = get_optimalized_thresholds(tm, None, data[0], indicator=indicator)
        tm = TechnicalAnalyst(prices=data[1]['close'].tolist())
        ma1, ma2 = tm.get_moving_average_crossover(data[1], timeperiod1=m1, timeperiod2=m2)
        signals = tm.get_MAC_sell_buy_signals(ma1, ma2, stoploss=stoploss, threshold=stoploss_threshold)
    else:
        for i in range(0, len(data)):
            if i > 30:
                close = data[i-30:i]
            else:
                close = data[:i+1]

            tm = TechnicalAnalyst(prices=close['close'].tolist())
            last = next(filter(lambda x: x != 0, reversed(signals)), 0)
            current_price = close['close'].tolist()[-1]

            if indicator == 'RSI':
                indicator_value = tm.get_relative_strength_index(close)
                final_lower_limit, final_upper_limit = get_optimalized_thresholds(tm, indicator_value, close, indicator=indicator)
                area.append('down' if indicator_value[-1] < final_lower_limit else 'top' if indicator_value[-1] > final_upper_limit else 'middle')

                if area[i-1] == 'down' and area[i] == 'middle' and last != 1:
                    signals.append(1)
                    bought_for = current_price
                elif area[i-1] == 'top' and area[i] == 'middle' and last != 2:
                    signals.append(2)
                else:
                    if stoploss and tm.check_if_exceeded_stop_loss(current_price, bought_for, threshold=stoploss_threshold) and last != 2:
                        signals.append(2)
                    else:
                        signals.append(0)
            elif indicator == 'ROC':
                indicator_value = tm.get_rate_of_change(close)
                final_limit = get_optimalized_thresholds(tm, indicator_value, close, indicator=indicator)
                area.append('down' if indicator_value[-1] < final_limit else 'top')

                if area[i-1] == 'down' and area[i] == 'top' and last != 1:
                    signals.append(1)
                    bought_for = current_price
                elif area[i-1] == 'top' and area[i] == 'down' and last != 2:
                    signals.append(2)
                else:
                    if stoploss and tm.check_if_exceeded_stop_loss(current_price, bought_for, threshold=stoploss_threshold) and last != 2:
                        signals.append(2)
                    else:
                        signals.append(0)
            elif indicator == 'CCI':
                indicator_value = tm.get_commodity_channel_index(close)
                final_lower_limit, final_upper_limit = get_optimalized_thresholds(tm, indicator_value, close, indicator=indicator)
                area.append('down' if indicator_value[-1] < final_lower_limit else 'top' if indicator_value[-1] > final_upper_limit else 'middle')

                if area[i-1] == 'middle' and area[i] == 'top' and last != 1:
                    signals.append(1)
                    bought_for = current_price
                elif area[i-1] == 'middle' and area[i] == 'down' and last != 2:
                    signals.append(2)
                else:
                    if stoploss and tm.check_if_exceeded_stop_loss(current_price, bought_for, threshold=stoploss_threshold) and last != 2:
                        signals.append(2)
                    else:
                        signals.append(0)
            elif indicator == 'SO':
                k, d = tm.get_stochastic_oscillator(close)
                final_lower_limit, final_upper_limit = get_optimalized_thresholds(tm, (k, d), close, indicator=indicator)
                
                k, d = k.tolist(), d.tolist()
                area.append('down' if k[-1] <= d[-1] else 'top')

                if k[-1] < final_lower_limit and d[-1] < final_lower_limit and area[i-1] == 'down' and area[i] == 'top' and last != 1:
                    signals.append(1)
                    bought_for = current_price
                elif k[-1] > final_upper_limit and d[-1] > final_upper_limit and area[i-1] == 'top' and area[i] == 'down' and last != 2:
                    signals.append(2)
                else:
                    if stoploss and tm.check_if_exceeded_stop_loss(current_price, bought_for, threshold=stoploss_threshold) and last != 2:
                        signals.append(2)
                    else:
                        signals.append(0)
            elif indicator == 'WO':
                indicator_value = tm.get_williams_oscillator(close)
                final_lower_limit, final_upper_limit = get_optimalized_thresholds(tm, indicator_value, close, indicator=indicator)
                indicator_value = indicator_value.tolist()
                area.append('down' if indicator_value[-1] < final_lower_limit else 'top' if indicator_value[-1] > final_upper_limit else 'middle')
                

                if area[i-1] == 'down' and area[i] == 'middle' and last != 1:
                    signals.append(1)
                    bought_for = current_price
                elif area[i-1] == 'top' and area[i] == 'middle' and last != 2:
                    signals.append(2)
                else:
                    if tm.check_if_exceeded_stop_loss(current_price, bought_for, threshold=stoploss_threshold) and last != 2 and stoploss:
                        signals.append(2)
                    else:
                        signals.append(0)
         
    return signals


# data = pd.read_csv(f"data/BTC.csv")
# print(get_oplimalized_signals(data))