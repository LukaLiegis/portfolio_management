import numpy as np
import polars as pl


def winsorize(data, percentile=0.05, axis=0):
    """
    Winsorize data to symmetric percentiles.
    """
    if not 0 <= percentile <= 1:
        raise ValueError('percentile must be between 0 and 1')

    fin_data = np.where(np.isfinite(data), data, np.nan)

    lower = np.nanpercentile(fin_data, percentile * 100, axis=axis, keepdims=True)
    upper = np.nanpercentile(fin_data, (1 - percentile) * 100, axis=axis, keepdims=True)

    return np.clip(data, lower, upper)


def center_xsection(target_col, over_col, standardize=False):
    """
    Cross-sectionally center and optionally standardize a column.
    """
    expr = pl.col(target_col) - pl.col(target_col).mean().over(over_col)

    if standardize:
        expr = expr / pl.col(target_col).std().over(over_col)

    return expr


def exp_weights(window, half_life):
    """
    Generate exponentially decaying weights.
    """
    decay = np.log(2) / half_life
    return np.exp(-decay * np.arange(window))[::-1]