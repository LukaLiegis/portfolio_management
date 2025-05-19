# biotech/backtest.py
import polars as pl
import numpy as np


def backtest_strategy(
        initial_capital,
        stock_data,
        factor_data,
        strategy_func,
        rebalance_frequency=21,  # Monthly in trading days
        transaction_cost=0.0005,  # 5bps per trade
        start_date=None,
        end_date=None
):
    """
    Backtest a factor-based strategy with realistic constraints.

    Parameters:
    -----------
    initial_capital: Starting portfolio value
    stock_data: Historical stock data
    factor_data: Factor returns data
    strategy_func: Function that generates portfolio positions
    rebalance_frequency: How often to rebalance (in trading days)
    transaction_cost: Cost per trade as fraction of value
    """
    # Setup backtest dates
    dates = stock_data.select("date").unique().sort("date")

    if start_date:
        dates = dates.filter(pl.col("date") >= start_date)
    if end_date:
        dates = dates.filter(pl.col("date") <= end_date)

    # Initialize backtest variables
    portfolio_value = initial_capital
    positions = {}
    portfolio_history = []

    # Run backtest
    for i, date in enumerate(dates.to_series()):
        print(f"Processing date: {date}")

        # Get data as of current date
        current_data = stock_data.filter(pl.col("date") <= date)
        current_factors = factor_data.filter(pl.col("date") <= date)

        # Rebalance on schedule or at start
        if i == 0 or i % rebalance_frequency == 0:
            # Generate new portfolio
            new_positions, portfolio_stats = strategy_func(
                current_data, current_factors, date
            )

            # Calculate transaction costs
            if positions:
                # Calculate position changes
                turnover = sum(abs(new_positions.get(ticker, 0) - positions.get(ticker, 0))
                               for ticker in set(new_positions) | set(positions))
                cost = turnover * transaction_cost
            else:
                # Initial positions
                turnover = sum(abs(pos) for pos in new_positions.values())
                cost = turnover * transaction_cost

            # Apply transaction costs
            portfolio_value -= cost
            positions = new_positions

        # Get daily returns
        next_date_idx = i + 1
        if next_date_idx < len(dates):
            next_date = dates[next_date_idx, 0]
            day_returns = stock_data.filter(pl.col("date") == next_date)

            # Calculate portfolio return
            portfolio_return = sum(
                positions.get(ticker, 0) *
                day_returns.filter(pl.col("symbol") == ticker)["asset_returns"].get(0, 0)
                for ticker in positions
            ) / portfolio_value

            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)

        # Record portfolio state
        portfolio_history.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "positions": positions.copy()
        })

    return portfolio_history