import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def create_factor_model(stocks, returns_data):
    '''
    Create simple factor model with market data.
    :param stocks:
    :param returns_data:
    :return:
    '''
    print('Creating factor model...')

    market_returns = returns_data['SPY']

    factor_betas = {}
    idio_vol = {}
    factor_r2 = {}

    for ticker in stocks:

        if ticker not in returns_data.columns:
            continue

        stock_returns = returns_data[ticker]

        valid_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(valid_data) < 0:
            continue

        X = valid_data.iloc[:, 1].values.reshape(-1, 1)
        y = valid_data.iloc[:, 0].values

        model = LinearRegression()
        model.fit(X, y)
        beta = model.coef_[0]
        alpha = model.intercept_
        residuals = y - model.predict(X)

        factor_betas[ticker] = beta
        idio_vol[ticker] = np.std(residuals)
        factor_r2[ticker] = model.score(X, y)

    factor_betas = pd.Series(factor_betas)
    idio_vol = pd.Series(idio_vol)
    factor_r2 = pd.Series(factor_r2)

    return factor_betas, idio_vol, factor_r2