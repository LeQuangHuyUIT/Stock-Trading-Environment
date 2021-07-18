
import pandas as pd
import numpy as np
#!pip install ta
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi
import seaborn as sns
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
#in colab
from env.utils import Plot_OHCL
# from utils import Plot_OHCL
from finta import TA

def AddIndicators(df):
    df_indicator = df.copy(deep= True)
    df_indicator['SMA'] = TA.SMA(df_indicator, 12)
    df_indicator[['TENKAN', 'KIJUN', 'senkou_span_a', 'SENKOU', 'CHIKOU']] = TA.ICHIMOKU(df)
    df_indicator['RSI'] = TA.RSI(df_indicator)
    df_indicator['EMA'] = TA.EMA(df_indicator)
    df_indicator = df_indicator.fillna(0)
    return df_indicator

if __name__ == "__main__":   
    df = pd.read_csv('/home/huyle/MyGit/Stock-Trading-Environment/data/vic_stock.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)


