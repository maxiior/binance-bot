import streamlit as st
from technical_analysis import TechnicalAnalyst
from api import API
from binance import Client
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from nn import _2FC
import torch
import datetime
import time
from sklearn.model_selection import train_test_split
from optimization import get_oplimalized_signals

def use_heuristics(data, initial_capital, stoploss=False, stoploss_threshold=0.03):
    tm = TechnicalAnalyst(prices=data['close'].tolist())

    rsi = tm.get_relative_strength_index(data)
    rsi_signals = tm.get_RSI_sell_buy_signals(rsi, stoploss=stoploss, threshold=stoploss_threshold)

    roc = tm.get_rate_of_change(data)
    roc_signals = tm.get_ROC_sell_buy_signals(roc, stoploss=stoploss, threshold=stoploss_threshold)

    cci = tm.get_commodity_channel_index(data)
    cci_signals = tm.get_CCI_sell_buy_signals(cci, stoploss=stoploss, threshold=stoploss_threshold)

    k, d = tm.get_stochastic_oscillator(data)
    so_signals = tm.get_SO_sell_buy_signals(k, d, stoploss=stoploss, threshold=stoploss_threshold)

    wo = tm.get_williams_oscillator(data)
    wo_signals = tm.get_WO_sell_buy_signals(wo, stoploss=stoploss, threshold=stoploss_threshold)

    sar_signals = tm.get_SAR_sell_buy_signals(data, stoploss=stoploss, threshold=stoploss_threshold)

    ma1, ma2 = tm.get_moving_average_crossover(data)
    mac5_20_signals = tm.get_MAC_sell_buy_signals(ma1, ma2, stoploss=stoploss, threshold=stoploss_threshold)

    ma1, ma2 = tm.get_moving_average_crossover(data, timeperiod1=10, timeperiod2=50)
    mac10_50_signals = tm.get_MAC_sell_buy_signals(ma1, ma2, stoploss=stoploss, threshold=stoploss_threshold)

    macd, macd_signal, _ = tm.get_moving_average_convergence_divergence(data)
    macd_signals = tm.get_MAC_sell_buy_signals(macd, macd_signal, stoploss=stoploss, threshold=stoploss_threshold)

    df = pd.DataFrame(columns=['indicator', 'final capital ($)', 'profit (%)', 'transactions', 'profit/transaction'])

    df.loc[len(df)] = tm.calculate_profit(data, rsi_signals, name='RSI', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, roc_signals, name='ROC', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, cci_signals, name='CCI', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, so_signals, name='SO', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, wo_signals, name='WO', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, sar_signals, name='SAR', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, mac5_20_signals, name='MAC_5_20', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, mac10_50_signals, name='MAC_10_50', cash=initial_capital)
    df.loc[len(df)] = tm.calculate_profit(data, macd_signals, name='MACD', cash=initial_capital)

    return df, {'rsi': rsi_signals, 'roc': roc_signals, 'cci': cci_signals, 'so': so_signals, 'wo': wo_signals, 'sar': sar_signals, 'mac_5_20': mac5_20_signals, 'mac_10_50': mac10_50_signals, 'macd': macd_signals}

def check_if_current_data(file_name):
    if os.path.isfile(os.path.join("./data", file_name)):
        mod_time = os.path.getmtime(os.path.join("./data", file_name))
        mod_date = datetime.datetime.fromtimestamp(mod_time).date()

        if mod_date == datetime.date.today():
            return True
        else:
            return False
    else:
        return False

if __name__ == '__main__':
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    client = Client(API_KEY, SECRET_KEY)
    api = API(client)

    rl_model = 'BNB'

    st.caption('using binance api')

    option = st.selectbox('Select your model', ('Heuristics', 'Optimization', 'Reinforcement Learning', 'Neural Network'))

    if option == 'Heuristics' or option == 'Neural Network' or option == 'Optimization':
        col1, col2, col3 = st.columns(3)

        with col1:
            symbol = st.text_input("Set crypto stock symbol (/BUSD)", value="BTC")
        with col2:
            time_period = st.text_input("Type a time period (days)", value="180")
        with col3:
            initial_capital = st.text_input("Initial capital ($)", value="1000000")


        stoploss = st.checkbox('I want to use stop-loss')
        if stoploss:
            stoploss_threshold = st.text_input("stop-loss (0-1)", value="0.03")
        else:
            stoploss_threshold = 0.03

        if option == 'Optimization':
            indicator = st.radio("Choose oscillator:", ['RSI', 'ROC', 'CCI', 'SO', 'WO', 'MAC'])

            if indicator == 'MAC':
                train_size = st.text_input("Set training size (0.01-0.99)", value="0.50")
    
    elif option == 'Reinforcement Learning':
        col1, col2 = st.columns(2)

        with col1:
            time_period = st.text_input("Type a time period (days)", value="180")
        with col2:
            initial_capital = st.text_input("Initial capital ($)", value="1000000")

        files = [i.replace('.zip', '') for i in os.listdir("./models")]
        symbol = st.radio("Choose trained model:", files)

    

    if st.button("Invest"):
        try:
            if check_if_current_data(f"{symbol}_{time_period}.csv"):
                data = pd.read_csv(f"data/{symbol}_{time_period}.csv")
            else:
                data = api.get_historical_klines(f"{symbol}BUSD", client.KLINE_INTERVAL_1DAY, days=int(time_period))
                data.to_csv(f"data/{symbol}_{time_period}.csv", index=False)

            # line_chart = st.line_chart(pd.DataFrame(data['close'].tolist(), columns=[f'{symbol}']))

            if option == 'Heuristics':
                df, signals = use_heuristics(data, float(initial_capital), stoploss=stoploss, stoploss_threshold=float(stoploss_threshold))
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
                start = time.time()

                if indicator == 'MAC':
                    len1 = int(len(data)*float(train_size))
                    train_data, test_data = data[:len1], data[len1:]
                    signals = get_oplimalized_signals((train_data, test_data), indicator=indicator, stoploss=stoploss, stoploss_threshold=float(stoploss_threshold))
                else:
                    signals = get_oplimalized_signals(data, indicator=indicator, stoploss=stoploss, stoploss_threshold=float(stoploss_threshold))
                
                stop = time.time()
                print(stop-start)
                
                if indicator == 'MAC':
                    tm = TechnicalAnalyst(prices=test_data['close'].tolist())
                    profit = tm.calculate_profit(test_data, signals, name=indicator, cash=float(initial_capital))
                    result = [(idx, i, signals[idx]) for idx, i in enumerate(test_data['close'].tolist()) if signals[idx] != 0]
                else:
                    tm = TechnicalAnalyst(prices=data['close'].tolist())
                    profit = tm.calculate_profit(data, signals, name=indicator, cash=float(initial_capital))
                    result = [(idx, i, signals[idx]) for idx, i in enumerate(data['close'].tolist()) if signals[idx] != 0]

                X = [i[0] for i in result]
                y = [i[1] for i in result]
                c = [i[2] for i in result]

                st.dataframe(pd.DataFrame({'final capital ($)': [profit[1]], 'profit (%)': [profit[2]], 'transactions': [profit[3]], 'profit/transaction': [profit[4]]}))

                st.header(indicator)
                if indicator == 'MAC':
                    df = pd.DataFrame(test_data['close'].tolist(), columns=[f'{symbol}'])
                else:
                    df = pd.DataFrame(data['close'].tolist(), columns=[f'{symbol}'])
                fig, ax = plt.subplots()
                ax.plot(df)
                ax.scatter(x=X, y=y, color=['green' if i == 1 else 'red' for i in c], s=50)
                st.pyplot(fig)

            elif option == 'Neural Network':
                model = _2FC(input_size=5)
                model.load_state_dict(torch.load(f'./nn/model7.pt'))

                X = data['close'].to_list()[:-1]
                y = data['close'][1:]
                X2 = []

                for i in range(len(X)):
                    if i+4 >= len(X):
                        break
                    X2 += [[float(X[i]), float(X[i+1]), float(X[i+2]), float(X[i+3]), float(X[i+4])]]

                seq = torch.stack([torch.tensor(i).float() for i in X2])
                results = []

                with torch.no_grad():
                    for seq_batch in seq.to('cpu'):
                        preds = model(seq_batch.unsqueeze(0).to('cpu'))
                        preds = preds.detach().cpu().numpy()
                        results.append(preds[0][0])

                capital = float(initial_capital)
                crypto = 0
                transactions = 0

                colors = []
                X_points = []
                y_points = []

                bought_for = 0

                for idx, (i, j) in enumerate(zip(X[4:], results)):
                    if j > i and crypto == 0:
                        crypto = np.round(capital/i, 10)
                        capital = 0
                        transactions += 1
                        colors.append('green')
                        X_points.append(idx+4)
                        y_points.append(i)
                        bought_for = i
                    elif j < i and crypto != 0:
                        capital = crypto * i
                        crypto = 0
                        transactions += 1
                        colors.append('red')
                        X_points.append(idx+4)
                        y_points.append(i)
                    if bought_for != 0 and stoploss and crypto != 0:
                        if 1 - np.round(i/bought_for, 2) > float(stoploss_threshold):
                            capital = crypto * i
                            crypto = 0
                            transactions += 1
                            colors.append('red')
                            X_points.append(idx+4)
                            y_points.append(i)

                if capital == 0:
                    transactions += 1
                    capital = crypto * X[-1]

                try:
                    scores = pd.DataFrame({'final capital': [capital], 'profit (%)': [(100*capital/float(initial_capital))-100], 
                                        'transactions': [transactions], 'profit/transaction': [(capital - float(initial_capital))/transactions if transactions != 0 else 0]})
                    st.dataframe(scores)
                except Exception as e:
                    print(e)
            
                df = pd.DataFrame(data['close'].tolist(), columns=[f'{symbol}'])
                fig, ax = plt.subplots()
                ax.plot(df)
                ax.scatter(x=X_points, y=y_points, color=colors, s=50)
                st.pyplot(fig)

            elif option == 'Reinforcement Learning':
                WINDOW_SIZE = 5

                data.rename(columns={"open time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                try:
                    data.drop('Unnamed: 0', axis=1, inplace=True)
                except:
                    pass
                data.drop('close time', axis=1, inplace=True)
                data.drop('quote asset volume', axis=1, inplace=True)
                data.drop('number of trades', axis=1, inplace=True)
                data.drop('tb base volume', axis=1, inplace=True)
                data.drop('tb quote volume', axis=1, inplace=True)

                env = gym.make('stocks-v0', df=data, frame_bound=(WINDOW_SIZE,int(time_period)), window_size=WINDOW_SIZE)
                obs = env.reset()
                model = A2C('MlpPolicy', env, verbose=1)
                model.load(f"models/{rl_model}.zip")

                model_profit = 1
                random_profit = 1

                while True:
                    obs = obs[np.newaxis, ...]
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    if done:
                        model_profit = info['total_profit']
                        break
                fig, ax = plt.subplots()
                plt.cla()
                env.render_all()
                st.pyplot(fig)

                env = gym.make('stocks-v0', df=data, frame_bound=(WINDOW_SIZE,int(time_period)), window_size=WINDOW_SIZE)
                state = env.reset()

                while True:
                    action = env.action_space.sample()
                    n_state, rewards, done, info = env.step(action)
                    if done:
                        random_profit = info['total_profit']
                        break
                
                df = pd.DataFrame({'random profit': [random_profit], 'model_profit': [model_profit], 'final capital': [np.round(model_profit*float(initial_capital), 2)]})
                st.dataframe(df)
                
        except:
            st.error("Something went wrong.")