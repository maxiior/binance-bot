import streamlit as st
from technical_analysis import TechnicalAnalyst
from api import API
from binance import Client
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def use_heuristics(data, initial_capital):
    tm = TechnicalAnalyst()

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

    sar_signals = tm.get_SAR_sell_buy_signals(data)

    ma1, ma2 = tm.get_moving_average_crossover(data)
    mac5_20_signals = tm.get_MAC_sell_buy_signals(ma1, ma2)

    ma1, ma2 = tm.get_moving_average_crossover(data, timeperiod1=10, timeperiod2=50)
    mac10_50_signals = tm.get_MAC_sell_buy_signals(ma1, ma2)

    macd, macd_signal, _ = tm.get_moving_average_convergence_divergence(data)
    macd_signals = tm.get_MAC_sell_buy_signals(macd, macd_signal)

    df = pd.DataFrame(columns=['indicator', 'final capital ($)', 'profit (%)', 'transactions', 'profit/transactions'])

    df.loc[len(df)] = tm.calculate_profit(data, rsi_signals, name='RSI', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, roc_signals, name='ROC', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, cci_signals, name='CCI', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, so_signals, name='SO', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, wo_signals, name='WO', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, sar_signals, name='SAR', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, mac5_20_signals, name='MAC_5_20', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, mac10_50_signals, name='MAC_10_50', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, macd_signals, name='MACD', cash=initial_capital)

    return df, {'rsi': rsi_signals, 'roc': roc_signals, 'cci': cci_signals, 'so': so_signals, 'wo': wo_signals, 'sar': sar_signals, 'mac5_20': mac5_20_signals, 'mac10_50': mac10_50_signals, 'macd': macd_signals}



if __name__ == '__main__':
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    client = Client(API_KEY, SECRET_KEY)
    api = API(client)

    st.caption('using binance api')

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input("Set crypto stock symbol (/BUSD)", value="BTC")
    with col2:
        time_period = st.text_input("Type a time period (days)", value="180")
    with col3:
        initial_capital = st.text_input("Initial capital ($)", value="1000000")


    option = st.selectbox('Select your model', ('Heuristics', 'Optimization', 'Reinforcement Learning'))

    if st.button("Invest"):
        try:
            data = api.get_historical_klines(f"{symbol}BUSD", client.KLINE_INTERVAL_1DAY, days=int(time_period))

            

            # line_chart = st.line_chart(pd.DataFrame(data['close'].tolist(), columns=[f'{symbol}']))

            if option == 'Heuristics':
                df, signals = use_heuristics(data, float(initial_capital))
                df_tmp = df
                df = df.style.apply(lambda x: ['background-color: #66ff33' if x['profit (%)']>0 else 'background-color: #ff3300' for _ in x], axis=1)
                st.dataframe(df)

                name = df_tmp.loc[df_tmp['profit (%)'].idxmax(), 'indicator']
                signals = signals[name.lower()]
                result = [(idx, i, signals[idx]) for idx, i in enumerate(data['close'].tolist()) if signals[idx] != 0]
                X = [i[0] for i in result]
                y = [i[1] for i in result]
                c = [i[2] for i in result]

                st.header(name)
                df = pd.DataFrame(data['close'].tolist(), columns=[f'{symbol}'])
                fig, ax = plt.subplots()
                ax.plot(df)
                ax.scatter(x=X, y=y, color=['green' if i == 1 else 'red' for i in c], s=50)
                st.pyplot(fig)
            elif option == 'Optimization':
                pass
            elif option == 'Reinforcement Learning':
                pass
        except:
            st.error("Something went wrong.")