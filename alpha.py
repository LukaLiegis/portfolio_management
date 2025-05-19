import numpy as np
import pandas as pd


def estimate_alphas(stocks, returns, lookback_days = 252):

    alphas = pd.Series(index=stocks)

    recent_returns = returns[-lookback_days:]

    for ticker in stocks:
        if ticker in recent_returns.columns:
            stock_returns = recent_returns[ticker]
            sharpe = stock_returns.mean() / stock_returns.std() * np.sqrt(252)
            alphas[ticker] = sharpe
        else:
            alphas[ticker] = 0

    if not alphas.empty and not alphas.isna().all():
        alphas = alphas - alphas.mean()

    return alphas