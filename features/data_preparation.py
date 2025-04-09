from finta import TA
import pandas as pd
from finta.utils import resample_calendar
from datetime import date, timedelta



# function to get ohlcv df from binance data
def extract_ohlcv_from_binance_data(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                     'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('date')
    ohlcv = df.drop(['open_time', 'volume', 'close_time', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'], axis=1)
    ohlcv = ohlcv.rename(columns={'quote_volume' : 'volume'})
    return ohlcv


# generator of dates in selected interval
def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)


# function to unite all df's for each date in one
def concatenate_dataframes(start_date, end_date):
    ohlcv = pd.DataFrame()
    for single_date in daterange(start_date, end_date):
        date = single_date.strftime("%Y-%m-%d")
        ohlcv_temp = extract_ohlcv_from_binance_data(f'./data/BTC/BTCUSDT-1s-{date}.csv')
        ohlcv = pd.concat([ohlcv, ohlcv_temp])
    ohlcv = resample_calendar(ohlcv, "1s")
    return ohlcv
