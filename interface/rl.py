import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DQN

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from api import API
from dotenv import load_dotenv
import os
from binance import Client

WINDOW_SIZE = 5
CRYPTO = 'DOT'

load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
client = Client(API_KEY, SECRET_KEY)

api = API(client)
data = api.get_historical_klines(f"BTCBUSD", client.KLINE_INTERVAL_1DAY, days=3000)
data.to_csv(f'data/{CRYPTO}.csv', index=False)


df = pd.read_csv(f'data/{CRYPTO}.csv')

df.rename(columns={"open time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
try:
    df.drop('Unnamed: 0', axis=1, inplace=True)
except:
    pass

df.drop('close time', axis=1, inplace=True)
df.drop('quote asset volume', axis=1, inplace=True)
df.drop('number of trades', axis=1, inplace=True)
df.drop('tb base volume', axis=1, inplace=True)
df.drop('tb quote volume', axis=1, inplace=True)
df.drop('ignore', axis=1, inplace=True)


# env = gym.make('stocks-v0', df=df, frame_bound=(WINDOW_SIZE,len(df)), window_size=WINDOW_SIZE)
# state = env.reset()

# while True:
#     action = env.action_space.sample()
#     n_state, rewards, done, info = env.step(action)
#     if done:
#         print("info", info)
#         break
# plt.figure(figsize=(15,6))
# plt.cla()
# env.render_all()
# plt.show()

env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(WINDOW_SIZE,len(df)), window_size=WINDOW_SIZE)
env = DummyVecEnv([env_maker])

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

env = gym.make('stocks-v0', df=df, frame_bound=(WINDOW_SIZE,len(df)), window_size=WINDOW_SIZE)
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

model.save(f"models/{CRYPTO}")