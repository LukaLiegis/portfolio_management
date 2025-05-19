import polars as pl
import numpy as np


def build_biotech_risk_model(stock_returns, factor_returns, mcaps):
    
    stock_returns_np = stock_returns.to_numpy()
    factor_returns_np = factor_returns.to_numpy()

    n_stocks = stock_returns_np.shape[0]
    n_factors = factor_returns_np.shape[1]

    exposures = np.zeros((n_stocks, n_factors))
    specific_returns = np.zeros_like(stock_returns_np)

    weights = np.sqrt(mcaps.to_numpy().reshape(-1, 1))

    # For each stock, estimate factor exposures
    for i in range(n_stocks):

        W = np.diag(weights[i, :])

        X = factor_returns_np
        y = stock_returns_np[i, :]

        exposures[i, :] = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]

        specific_returns[i, :] = y - X @ exposures[i, :]

    factor_cov = np.cov(factor_returns_np.T)

    specific_var = np.var(specific_returns, axis=1)

    return exposures, factor_cov, specific_var