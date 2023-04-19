import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from api import API
from dotenv import load_dotenv
import os
from binance import Client

load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
client = Client(API_KEY, SECRET_KEY)

api = API()
data = api.get_historical_klines(f"BTCBUSD", client.KLINE_INTERVAL_1DAY, days=180)