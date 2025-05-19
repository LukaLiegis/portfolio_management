import polars as pl
import numpy as np
from math_utils import exp_weights, center_xsection, winsorize


def factor_mom(returns_df, trailing_days=252, half_life=126, lag=20):
    """
    Create momentum factor scores.
    """
    weights = exp_weights(trailing_days, half_life)

    def weighted_momentum(values):
        return (np.cumprod(1 + (values * weights[-len(values):])) - 1)[-1]

    # Group by ticker and calculate rolling momentum
    result = (
        returns_df
        .sort("date")
        .with_columns(pl.col("asset_returns").shift(lag).over("symbol").alias("lagged_returns"))
        .with_columns(
            pl.col("lagged_returns")
            .rolling_map(weighted_momentum, window_size=trailing_days)
            .over("symbol")
            .alias("mom_score")
        )
    )

    # Cross-sectionally center and standardize
    result = result.with_columns(
        center_xsection("mom_score", "date", True).alias("mom_score")
    )

    return result.select("date", "symbol", "mom_score")


def factor_size(stock_data):
    """
    Create size factor scores for each stock.
    """
    # Use market cap as size factor, inverted to match SMB
    return (
        stock_data
        .with_columns(
            pl.col("market_cap").log().alias("size_raw") * -1
        )
        .with_columns(
            center_xsection("size_raw", "date", True).alias("size_score")
        )
        .select("date", "symbol", "size_score")
    )


def factor_value(stock_data):
    """
    Create value factor scores.
    """
    value_df = (
        stock_data
        .with_columns([
            # Create value metrics from available data
            (pl.col("eps") / pl.col("prccd")).alias("earnings_price"),
            (pl.col("div") / pl.col("prccd")).alias("div_yield")
        ])
        .with_columns([
            # Cross-sectionally standardize
            center_xsection("earnings_price", "date", True).alias("ep_score"),
            center_xsection("div_yield", "date", True).alias("dy_score")
        ])
        .with_columns(
            # Combine into composite value score
            ((pl.col("ep_score") + pl.col("dy_score")) / 2).alias("value_score")
        )
        .select("date", "symbol", "value_score")
    )

    return value_df


def factor_quality(stock_data, returns_data):
    """
    Create quality factor.
    """
    # Calculate volatility using available returns data
    volatility = (
        returns_data
        .group_by("symbol")
        .agg(pl.col("asset_returns").std().alias("volatility"))
    )

    # Calculate turnover from your available data
    turnover = (
        stock_data
        .with_columns(
            (pl.col("cshtrd") / pl.col("cshoc")).alias("turnover")
        )
        .group_by(["date", "symbol"])
        .agg(pl.col("turnover").mean())
    )

    # Combine metrics into quality score
    quality = (
        turnover
        .join(volatility, on="symbol")
        .with_columns(
            # Higher turnover and lower volatility = higher quality
            (pl.col("turnover") - pl.col("volatility")).alias("quality_raw")
        )
        .with_columns(
            center_xsection("quality_raw", "date", True).alias("quality_score")
        )
        .select("date", "symbol", "quality_score")
    )

    return quality