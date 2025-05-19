import polars as pl
import numpy as np
from datetime import datetime


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

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates_df = stock_data.select("date").unique().sort("date")
    dates = dates_df["date"].to_list()

    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    if not dates:
        raise ValueError("No dates available for backtesting within the specified range")

    portfolio_value = initial_capital
    positions = {}
    portfolio_history = []
    turnover_history = []
    returns_history = []

    print(f"Starting backtest from {dates[0]} to {dates[-1]}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Rebalance frequency: {rebalance_frequency} trading days")
    print(f"Transaction cost: {transaction_cost * 10000:.1f} bps")

    # Run backtest
    for i, date in enumerate(dates):
        current_date_str = date.strftime("%Y-%m-%d")
        print(f"Processing date: {current_date_str} ({i + 1}/{len(dates)})", end="\r")

        # Get data as of current date
        current_stocks = stock_data.filter(pl.col("date") <= date)
        current_factors = factor_data.filter(pl.col("date") <= date)

        available_symbols = (current_stocks
                             .filter(pl.col("date") == date)
                             .select("symbol")
                             .unique()["symbol"]
                             .to_list())

        should_rebalance = (i == 0) or (i % rebalance_frequency == 0)

        if should_rebalance:
            try:
                new_positions, portfolio_stats = strategy_func(
                    current_stocks, current_factors, date
                )

                if positions:
                    turnover = 0
                    for ticker in set(new_positions) | set(positions):
                        old_pos = positions.get(ticker, 0)
                        new_pos = new_positions.get(ticker, 0)
                        turnover += abs(new_pos - old_pos)

                    cost = turnover * transaction_cost
                else:
                    turnover = sum(abs(pos) for pos in new_positions.values())
                    cost = turnover * transaction_cost

                turnover_history.append({
                    'date': date,
                    'turnover': turnover / portfolio_value,
                    'cost': cost
                })

                portfolio_value -= cost
                positions = new_positions

                portfolio_history.append({
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "positions": positions.copy(),
                    "is_rebalance": True,
                    "portfolio_stats": portfolio_stats
                })

                print(f"\nRebalanced portfolio on {current_date_str}. GMV: ${portfolio_value:,.2f}")
                print(f"Transaction cost: ${cost:,.2f} ({cost / portfolio_value * 100:.2f}%)")

            except Exception as e:
                print(f"\nError during rebalance on {current_date_str}: {str(e)}")
                # If rebalance fails but we already have positions, continue with existing positions
                if not positions:
                    raise RuntimeError("Failed to generate initial portfolio")

                portfolio_history.append({
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "positions": positions.copy(),
                    "is_rebalance": False,
                    "error": str(e)
                })
        else:
            # Record portfolio state without rebalancing
            portfolio_history.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "positions": positions.copy(),
                "is_rebalance": False
            })

        next_date_idx = i + 1
        if next_date_idx < len(dates):
            next_date = dates[next_date_idx]

            # Get returns for the next day
            next_day_data = stock_data.filter(pl.col("date") == next_date)

            if len(next_day_data) > 0:
                # Calculate portfolio return
                portfolio_return = 0

                for ticker, position in positions.items():
                    # Find next day's return for this stock
                    stock_data_row = next_day_data.filter(pl.col("symbol") == ticker)

                    if len(stock_data_row) > 0 and "asset_returns" in stock_data_row.columns:
                        stock_return = stock_data_row["asset_returns"][0]

                        if np.isfinite(stock_return):

                            portfolio_return += (position / portfolio_value) * stock_return

                returns_history.append({
                    'date': next_date,
                    'return': portfolio_return
                })

                portfolio_value *= (1 + portfolio_return)

    print(f"\nBacktest completed. Final portfolio value: ${portfolio_value:,.2f}")

    returns_df = pl.DataFrame(returns_history)

    if len(returns_df) > 0:

        cum_returns = np.cumprod(1 + returns_df["return"].to_numpy()) - 1

        peak = np.maximum.accumulate(cum_returns + 1) - 1
        drawdowns = (cum_returns - peak) / (peak + 1)

        rolling_window = min(63, len(returns_df) // 2)

        if len(returns_df) > rolling_window:
            rolling_returns = [
                np.prod(1 + returns_df["return"].to_numpy()[i:i + rolling_window]) - 1
                for i in range(len(returns_df) - rolling_window)
            ]

            rolling_sharpe = [
                r / (np.std(returns_df["return"].to_numpy()[i:i + rolling_window]) * np.sqrt(252))
                if np.std(returns_df["return"].to_numpy()[i:i + rolling_window]) > 0 else 0
                for i, r in enumerate(rolling_returns)
            ]
        else:
            rolling_sharpe = []

        total_days = (dates[-1] - dates[0]).days
        years = total_days / 365.25

        annualized_return = (1 + cum_returns[-1]) ** (1 / years) - 1
        annualized_volatility = np.std(returns_df["return"].to_numpy()) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        max_drawdown = np.min(drawdowns)

        backtest_results = {
            'initial_capital': initial_capital,
            'final_value': portfolio_value,
            'absolute_return': portfolio_value - initial_capital,
            'return_pct': (portfolio_value / initial_capital) - 1,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': np.mean([t['turnover'] for t in turnover_history]) if turnover_history else 0,
            'dates': dates,
            'portfolio_history': portfolio_history,
            'returns': returns_df["return"].to_numpy(),
            'cumulative_returns': cum_returns,
            'drawdowns': drawdowns,
            'rolling_sharpe': rolling_sharpe
        }
    else:
        backtest_results = {
            'initial_capital': initial_capital,
            'final_value': portfolio_value,
            'absolute_return': portfolio_value - initial_capital,
            'return_pct': (portfolio_value / initial_capital) - 1,
            'portfolio_history': portfolio_history,
        }

    return backtest_results