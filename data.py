import pandas as pd
import yfinance as yf

from config import STOCKS, BENCHMARKS

def get_data(start_date, end_date):
    '''
    Fetch historical data for stocks and benchmarks.
    :param ticker:
    :return:
    '''
    print('Fetching historical data...')

    all_tickers = STOCKS + BENCHMARKS

    df = yf.download(tickers=all_tickers, start_date = start_date, end_date = end_date)['Close']
    price_data = df.fillna(method='ffill')

    returns_data = price_data.pct_change().dropna()

    momentum_end = pd.to_datetime(end_date) - pd.DateOffset(months=1)
    momentum_start = pd.to_datetime(end_date) - pd.DateOffset(months=13)

    try:
        mom_data = yf.download(tickers=all_tickers, start=momentum_start.strftime('%Y-%m-%d'),
                               end=momentum_end.strftime('%Y-%m-%d'))['Close']
        momentum = mom_data.iloc[-1]/ mom_data.iloc[0] - 1
    except:
        momentum = pd.Series(0, index=all_tickers)

    return price_data, returns_data, momentum