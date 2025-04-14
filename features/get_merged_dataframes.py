import pandas as pd
from finta import TA
from sklearn.model_selection import train_test_split

def strip_arrays(data1, data2):
    start_max = max(data1['date'].iloc[0], data2['date'].iloc[0])
    end_min = min(data1['date'].iloc[-1], data2['date'].iloc[-1])
    print(f"start: {start_max}, end: {end_min}")
    
    data1 = data1[(data1['date'] >= start_max) & (data1['date'] <= end_min)]
    data2 = data2[(data2['date'] >= start_max) & (data2['date'] <= end_min)]
    return data1, data2

def generate_indicators(df):
    # # short-term (_st)
    # moving averages
    df.loc[:, 'prev_change_st'] = (df['close'] - df['close'].shift(1)) / df['close']
    df.loc[:, 'SMA_3m_st'] = TA.SMA(df, period=3)
    df.loc[:, 'SMA_10m_st'] = TA.SMA(df, period=10)
    df.loc[:, 'SMA_1h_st'] = TA.SMA(df, period=60)

    df.loc[:, 'EMA_5m_st'] = TA.EMA(df, period=5)
    df.loc[:, 'EMA_60m_st'] = TA.EMA(df, period=60)

    # oscillator
    df.loc[:, 'ATR_5'] = TA.ATR(df, period=5)
    df.loc[:, 'ATR_11'] = TA.ATR(df, period=11)
    df.loc[:, 'ATR_21'] = TA.ATR(df, period=21)
    df.loc[:, 'ATR_81'] = TA.ATR(df, period=81)
    
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


def generate_indicators_multitimeframe(df):
    """
    Generate a wide range of technical indicators using the finta (TA) library.
    Each indicator is computed with three lookback setups:
      - ST : Short-Term
      - MT : Mid-Term
      - LT : Long-Term
    Column names are prefixed by an indicator type (e.g., MA_, OSC_, MOM_, BOL_, VOL_, TREND_)
    and suffixed with _ST, _MT, or _LT so that you can later filter based on the time horizon.
    
    Notes:
      - For volume-based indicators, finta provides ADL and CHAIKIN. Since those functions do not 
        accept period parameters, we mimic multi-timeframe features by applying rolling averages.
      - BBANDS returns a DataFrame with columns "BB_UPPER", "BB_MIDDLE", and "BB_LOWER".
      - MACD returns two columns ("MACD" and "SIGNAL") but no oscillator column.
      - For stochastic indicators, we use:
          * STOCH: returns %K,
          * STOCHD: returns %D (a rolling average of %K),
          * STOCHRSI: applies the stochastic formulation on RSI.
    
    Assumes the DataFrame has the following columns:
      'open', 'high', 'low', 'close', 'volume'
    """
    import numpy as np
    import pandas as pd
    from finta import TA

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # -------------------------
    # RAW Features
    # -------------------------
    df['RAW_prev_return'] = df['close'].pct_change()

    # -------------------------
    # MA - Moving Averages
    # -------------------------
    # Simple Moving Averages (SMA)
    df['MA_SMA_ST']       = TA.SMA(df, 3)         # very short-term SMA
    df['MA_SMA_MT']       = TA.SMA(df, 50)        # mid-term SMA
    df['MA_SMA_LT']       = TA.SMA(df, 200)       # long-term SMA
    df['MA_SMA_daily_LT'] = TA.SMA(df, 1440)      # daily average (for minute data)

    # Exponential Moving Averages (EMA)
    df['MA_EMA_ST']       = TA.EMA(df, 5)
    df['MA_EMA_MT']       = TA.EMA(df, 50)
    df['MA_EMA_LT']       = TA.EMA(df, 200)
    df['MA_EMA_daily_LT'] = TA.EMA(df, 1440)

    # -------------------------
    # OSC - Oscillators
    # -------------------------
    # Relative Strength Index (RSI)
    df['OSC_RSI_ST'] = TA.RSI(df, 7)
    df['OSC_RSI_MT'] = TA.RSI(df, 14)
    df['OSC_RSI_LT'] = TA.RSI(df, 50)

    # MACD: finta returns "MACD" and "SIGNAL"
    # Short-Term MACD:
    macd_st = TA.MACD(df, 3, 10, 6)
    df['OSC_MACD_ST']        = macd_st['MACD']
    df['OSC_MACD_SIGNAL_ST'] = macd_st['SIGNAL']
    
    # Mid-Term MACD:
    macd_mt = TA.MACD(df, 12, 26, 9)
    df['OSC_MACD_MT']        = macd_mt['MACD']
    df['OSC_MACD_SIGNAL_MT'] = macd_mt['SIGNAL']
    
    # Long-Term MACD:
    macd_lt = TA.MACD(df, 60, 240, 30)
    df['OSC_MACD_LT']        = macd_lt['MACD']
    df['OSC_MACD_SIGNAL_LT'] = macd_lt['SIGNAL']
    
    # --- Stochastic Indicators ---
    # Using finta’s STOCH, STOCHD, and STOCHRSI, where:
    #   STOCH returns %K,
    #   STOCHD returns %D (3-period moving average of %K),
    #   STOCHRSI computes the stochastic RSI.
    #
    # Short-Term stochastic: using period=5 (example)
    df['OSC_STOCH_K_ST']   = TA.STOCH(df, period=5)
    df['OSC_STOCH_D_ST']   = TA.STOCHD(df, period=3, stoch_period=5)
    df['OSC_STOCH_RSI_ST'] = TA.STOCHRSI(df, rsi_period=5, stoch_period=5)
    
    # Mid-Term stochastic: using period=14 (example)
    df['OSC_STOCH_K_MT']   = TA.STOCH(df, period=14)
    df['OSC_STOCH_D_MT']   = TA.STOCHD(df, period=3, stoch_period=14)
    df['OSC_STOCH_RSI_MT'] = TA.STOCHRSI(df, rsi_period=14, stoch_period=14)
    
    # Long-Term stochastic: using period=50 (example)
    df['OSC_STOCH_K_LT']   = TA.STOCH(df, period=50)
    df['OSC_STOCH_D_LT']   = TA.STOCHD(df, period=3, stoch_period=50)
    df['OSC_STOCH_RSI_LT'] = TA.STOCHRSI(df, rsi_period=50, stoch_period=50)
    
    # Money Flow Index (MFI)
    df['OSC_MFI_ST'] = TA.MFI(df, 14)
    df['OSC_MFI_MT'] = TA.MFI(df, 28)
    df['OSC_MFI_LT'] = TA.MFI(df, 50)

    # -------------------------
    # MOM - Momentum / Volatility
    # -------------------------
    # Average True Range (ATR)
    df['MOM_ATR_ST'] = TA.ATR(df, 14)
    df['MOM_ATR_MT'] = TA.ATR(df, 50)
    df['MOM_ATR_LT'] = TA.ATR(df, 100)
    
    # Average Directional Index (ADX)
    df['MOM_ADX_ST'] = TA.ADX(df, 14)
    df['MOM_ADX_MT'] = TA.ADX(df, 50)
    df['MOM_ADX_LT'] = TA.ADX(df, 100)
    
    # Rate of Change (ROC)
    df['MOM_ROC_ST'] = TA.ROC(df, 14)
    df['MOM_ROC_MT'] = TA.ROC(df, 50)
    df['MOM_ROC_LT'] = TA.ROC(df, 100)
    
    # -------------------------
    # BOL - Bollinger Bands
    # -------------------------
    # BBANDS returns columns "BB_UPPER", "BB_MIDDLE", "BB_LOWER"
    bb_st = TA.BBANDS(df, period=10)
    df['BOL_BB_UPPER_ST']  = bb_st['BB_UPPER']
    df['BOL_BB_MIDDLE_ST'] = bb_st['BB_MIDDLE']
    df['BOL_BB_LOWER_ST']  = bb_st['BB_LOWER']
    df['BOL_width_ST']     = bb_st['BB_UPPER'] - bb_st['BB_LOWER']
    
    bb_mt = TA.BBANDS(df, period=20)
    df['BOL_BB_UPPER_MT']  = bb_mt['BB_UPPER']
    df['BOL_BB_MIDDLE_MT'] = bb_mt['BB_MIDDLE']
    df['BOL_BB_LOWER_MT']  = bb_mt['BB_LOWER']
    df['BOL_width_MT']     = bb_mt['BB_UPPER'] - bb_mt['BB_LOWER']
    
    bb_lt = TA.BBANDS(df, period=50)
    df['BOL_BB_UPPER_LT']  = bb_lt['BB_UPPER']
    df['BOL_BB_MIDDLE_LT'] = bb_lt['BB_MIDDLE']
    df['BOL_BB_LOWER_LT']  = bb_lt['BB_LOWER']
    df['BOL_width_LT']     = bb_lt['BB_UPPER'] - bb_lt['BB_LOWER']

    # -------------------------
    # VOL - Volume-Based Indicators
    # -------------------------
    # ADL and CHAIKIN do not take period parameters. We mimic multiple timeframes with rolling averages.
    adl = TA.ADL(df)
    df['VOL_ADL_ST']     = adl.rolling(window=10).mean()
    df['VOL_ADL_MT']     = adl.rolling(window=50).mean()
    df['VOL_ADL_LT']     = adl.rolling(window=200).mean()
    
    chaikin = TA.CHAIKIN(df)
    df['VOL_CHAIKIN_ST'] = chaikin.rolling(window=10).mean()
    df['VOL_CHAIKIN_MT'] = chaikin.rolling(window=50).mean()
    df['VOL_CHAIKIN_LT'] = chaikin.rolling(window=200).mean()
    
    # -------------------------
    # TREND - Trend Indicators
    # -------------------------
    # Commodity Channel Index (CCI)
    df['TREND_CCI_ST'] = TA.CCI(df, 20)
    df['TREND_CCI_MT'] = TA.CCI(df, 50)
    df['TREND_CCI_LT'] = TA.CCI(df, 100)
    
    # # Parabolic SAR (PSAR) computed once (no period parameter)
    # original_index = df.index
    # df_reset = df.reset_index(drop=True)
    # psar_df = TA.PSAR(df_reset)  # Returns a DataFrame with columns like "psar", "psarbull", "psarbear"
    # # Restore the original index
    # psar_df.index = original_index
    # df['TREND_PSAR'] = psar_df['psar']
    
    return df


def get_merged_dataframes(pair1, pair2, generate_features_func=generate_indicators_multitimeframe):
    data1 = pd.read_feather(f'../ft_userdata/user_data/data/binance/{pair1}-1m.feather')
    data2 = pd.read_feather(f'../ft_userdata/user_data/data/binance/{pair2}-1m.feather')
    data1, data2 = strip_arrays(data1, data2)
    data1 = generate_features_func(data1)
    data2 = generate_features_func(data2)
    merged_df = pd.merge(data1, data2, on='date')
    df, test_df = train_test_split(merged_df, test_size=0.15, shuffle=False)
    train_df, valid_df = train_test_split(df, test_size=0.20, shuffle=False)
    return train_df, valid_df, test_df