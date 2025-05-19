import numpy as np
import pandas as pd
import polars as pl

def winsorize(
        data: np.ndarray,
        percentile: float = 0.05,
        axis: int = 0
) -> np.ndarray:
    """
    Winsorize each vector of a numpy array to symmetric percentiles.
    :param data: numpy array containing original data.
    :param percentile: the percentile at which to winsorize.
    :param axis: which axis to apply winsorization to.
    :return: numpy array containing winsorized data.
    """
    if not 0 <= percentile <= 1:
        raise ValueError('percentile must be between 0 and 1')

    fin_data = np.where(np.isfinite(data), data, np.nan)

    # Compute lower and upper percentiles
    lower_bounds = np.nanpercentile(fin_data, percentile * 100, axis=axis, keepdims=True)
    upper_bounds = np.nanpercentile(fin_data, (1 - percentile) * 100, axis=axis, keepdims=True)

    return np.clip(data, lower_bounds, upper_bounds)

def center_and_scale(
        df: pd.DataFrame,
        group_col: str,
        value_cols: list,
        standardize: bool = False
) -> pd.DataFrame:
    """
    Cross-sectionally center and optionally standardize columns.
    :param df: DataFrame to process.
    :param group_col: Column name to group by.
    :param value_cols: Column(s) to center/standardize.
    :param standardize: If true, standardize to unuit variance.
    :return: DataFrame with processed columns.
    """
    result = df.copy()

    for col in value_cols:
        grouped = result.groupby(group_col)[col]

        result[col] = result[col] - grouped.transform('mean')

        if standardize:
            result[col] = result[col] / grouped.transform('std')

    return result

def exp_weights(
        window: int,
        half_life: int
) -> np.ndarray:
    """
    Generate exponentially decaying weights.
    :param window: Number of points in lookback period.
    :param half_life: Innteger decay rate.
    :return: numpy arra of weights.
    """
