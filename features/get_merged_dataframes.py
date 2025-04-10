import numpy as np
import pandas as pd
import catboost
from datetime import date
from data_preparation import extract_ohlcv_from_binance_data, concatenate_dataframes
from finta import TA
import logging
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def strip_arrays(data1, data2):
    start_max = max(data1['date'].iloc[0], data2['date'].iloc[0])
    end_min = min(data1['date'].iloc[-1], data2['date'].iloc[-1])
    print(f"start: {start_max}, end: {end_min}")
    
    data1 = data1[(data1['date'] >= start_max) & (data1['date'] <= end_min)]
    data2 = data2[(data2['date'] >= start_max) & (data2['date'] <= end_min)]
    return data1, data2


def generate_indicators(df):
    df.loc[:, 'prev_change'] = (df['close'] - df['close'].shift(1)) / df['close']
    df.loc[:, 'SMA_3s'] = TA.SMA(df, period=3)
    df.loc[:, 'SMA_10s'] = TA.SMA(df, period=10)
    df.loc[:, 'SMA_20s'] = TA.SMA(df, period=20)
    df.loc[:, 'SMA_50s'] = TA.SMA(df, period=50)
    df.loc[:, 'SMA_100s'] = TA.SMA(df, period=100)
    df.loc[:, 'SMA_200s'] = TA.SMA(df, period=200)

    df.loc[:, 'EMA_3s'] = TA.EMA(df, period=5)
    df.loc[:, 'EMA_10s'] = TA.EMA(df, period=15)
    df.loc[:, 'EMA_20s'] = TA.EMA(df, period=25)
    df.loc[:, 'EMA_50s'] = TA.EMA(df, period=65)
    df.loc[:, 'EMA_100s'] = TA.EMA(df, period=130)
    df.loc[:, 'EMA_200s'] = TA.EMA(df, period=250)
    
    macd = TA.MACD(df, period_fast=3, period_slow=10, signal=6)
    df.loc[:, 'MACD_3_10s'] = macd['MACD']
    df.loc[:, 'SIGNAL_3_10s'] = macd['SIGNAL'] 
    
    # classic
    macd = TA.MACD(df, period_fast=12, period_slow=26, signal=9)
    df.loc[:, 'MACD_12_26s'] = macd['MACD']
    df.loc[:, 'SIGNAL_12_26s'] = macd['SIGNAL'] 
    
    macd = TA.MACD(df, period_fast=21, period_slow=45, signal=16)
    df.loc[:, 'MACD_21_45s'] = macd['MACD']
    df.loc[:, 'SIGNAL_21_45s'] = macd['SIGNAL'] 
    
    macd = TA.MACD(df, period_fast=98, period_slow=180, signal=30)
    df.loc[:, 'MACD_98_180s'] = macd['MACD']
    df.loc[:, 'SIGNAL_98_180s'] = macd['SIGNAL'] 
    
    df.loc[:, 'RSI_7s'] = TA.RSI(df, period=7)
    df.loc[:, 'RSI_14s'] = TA.RSI(df, period=14)
    df.loc[:, 'RSI_28s'] = TA.RSI(df, period=28)
    df.loc[:, 'RSI_56s'] = TA.RSI(df, period=56)
    
    bbands = TA.BBANDS(df, period=15, std_multiplier=2)
    df.loc[:, 'BBANDS_15_2_upper_diff'] = bbands['BB_UPPER'] - bbands['BB_MIDDLE'] # if negative the asset is overbought (price will go down)
    df.loc[:, 'BBANDS_15_2_lower_diff'] = bbands['BB_MIDDLE'] - bbands['BB_LOWER'] # if negative the asset is oversold (price will go up)
    bbands = TA.BBANDS(df, period=30, std_multiplier=2)
    df.loc[:, 'BBANDS_30_2_upper_diff'] = bbands['BB_UPPER'] - bbands['BB_MIDDLE'] # if negative the asset is overbought (price will go down)
    df.loc[:, 'BBANDS_30_2_lower_diff'] = bbands['BB_MIDDLE'] - bbands['BB_LOWER'] # if negative the asset is oversold (price will go up)
    
    df.loc[:, 'STOCH_5'] = TA.STOCH(df, period=5)
    df.loc[:, 'STOCH_15'] = TA.STOCH(df, period=15)
    df.loc[:, 'STOCH_50'] = TA.STOCH(df, period=50)
    
    df.loc[:, 'ATR_5'] = TA.ATR(df, period=5)
    df.loc[:, 'ATR_11'] = TA.ATR(df, period=11)
    df.loc[:, 'ATR_21'] = TA.ATR(df, period=21)
    df.loc[:, 'ATR_81'] = TA.ATR(df, period=81)
    
    df.loc[:, 'CCI_50'] = TA.CCI(df, period=50, constant=0.015)
    df.loc[:, 'CCI_100'] = TA.CCI(df, period=100, constant=0.015) # CCI > 100 => overbought, < -100 => oversold
    df.loc[:, 'CCI_200'] = TA.CCI(df, period=200, constant=0.015)
    
    df.loc[:, 'ROC_10'] = TA.CCI(df, period=10)
    df.loc[:, 'ROC_30'] = TA.CCI(df, period=30)
    df.loc[:, 'ROC_52'] = TA.CCI(df, period=52)
    df.loc[:, 'ROC_99'] = TA.CCI(df, period=99)
    
    df.loc[:, 'ADL'] = TA.ADL(df).fillna(0)
    
    df.loc[:, 'CHAIKIN'] = TA.CHAIKIN(df)
    return df

def get_merged_dataframes(pair1, pair2):
    data1 = pd.read_feather(f'../ft_userdata/user_data/data/binance/{pair1}-1m.feather')
    data2 = pd.read_feather(f'../ft_userdata/user_data/data/binance/{pair2}-1m.feather')
    data1, data2 = strip_arrays(data1, data2)
    data1 = generate_indicators(data1)
    data2 = generate_indicators(data2)
    merged_df = pd.merge(data1, data2, on='date')
    df, test_df = train_test_split(merged_df, test_size=0.15, shuffle=False)
    train_df, valid_df = train_test_split(df, test_size=0.20, shuffle=False)\
    return train_df, valid_df, test_df