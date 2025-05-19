import pandas as pd


def backtest_portfolio(portfolio, returns_data, benchmark: str, lookback_days: int = 252):
    '''
    Simple backtest of portfolio performance

    Parameters:
    -----------
    portfolio : DataFrame
        Portfolio with positions and weights
    returns_data : DataFrame
        Historical returns
    benchmark : str
        Benchmark ticker
    lookback_days : int
        Number of days for backtest

    Returns:
    --------
    tuple: (backtest_stats, backtest_returns)
    '''

    if portfolio is None or len(portfolio) == 0:
        print("No portfolio to backtest")
        return None, None

    backtest_returns = returns_data.iloc[-lookback_days:]

    portfolio_returns = pd.Series(index = backtest_returns.index)

    