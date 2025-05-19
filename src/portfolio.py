import polars as pl
import numpy as np
import cvxpy as cp


def construct_portfolio(
        alphas,
        factor_exposures,
        factor_covariance,
        specific_risk,
        target_gmv: int = 1_000_000,
        max_position: float = 0.15,
        max_vol: float = 0.15,  # Adding the missing parameter
        risk_aversion: float = 1.0,
        factor_constraints=None
):
    """
    Construct an optimal biotech portfolio with sophisticated risk controls.

    Uses convex optimization to maximize alpha subject to risk constraints.
    """
    n_stocks = len(alphas)

    # Default factor constraints if none provided
    if factor_constraints is None:
        factor_constraints = {
            'mktrf': 0.2,  # Market (±20%)
            'smb': 0.3,  # Size (±30%)
            'hml': 0.3,  # Value (±30%)
            'rmw': 0.3,  # Profitability (±30%)
            'cma': 0.3,  # Investment (±30%)
            'umd': 0.3  # Momentum (±30%)
        }

    # Setup optimization variables and constraints
    weights = cp.Variable(n_stocks)

    # Objective: maximize alpha (expected return)
    objective = cp.Maximize(alphas @ weights)

    # Risk calculations
    factor_risk = cp.quad_form(factor_exposures @ weights, factor_covariance)
    specific_risk_contrib = cp.sum(cp.multiply(cp.square(weights), specific_risk))

    # Constraints
    constraints = [
        # Budget constraint
        cp.sum(cp.abs(weights)) == 1,

        # Position size constraints
        weights >= -max_position,
        weights <= max_position,
    ]

    # Add factor exposure constraints
    for i, factor in enumerate(factor_constraints.keys()):
        max_exposure = factor_constraints[factor]
        factor_exposure = factor_exposures[:, i] @ weights
        constraints.extend([
            factor_exposure >= -max_exposure,
            factor_exposure <= max_exposure
        ])

    # Total risk constraint (optional)
    constraints.append(cp.sqrt(factor_risk + specific_risk_contrib) <= max_vol)

    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != 'optimal':
        print(f"Warning: Optimization problem status: {problem.status}")

    # Scale to target GMV
    positions = weights.value * target_gmv

    # Calculate portfolio statistics
    portfolio_stats = {
        'gmv': target_gmv,
        'expected_return': alphas @ weights.value,
        'factor_exposures': factor_exposures.T @ weights.value,
        'factor_risk': np.sqrt(
            weights.value.T @ factor_exposures @ factor_covariance @ factor_exposures.T @ weights.value),
        'specific_risk': np.sqrt(np.sum(weights.value ** 2 * specific_risk)),
        'total_risk': np.sqrt(
            weights.value.T @ factor_exposures @ factor_covariance @ factor_exposures.T @ weights.value +
            np.sum(weights.value ** 2 * specific_risk)
        )
    }

    return positions, portfolio_stats


def calculate_returns(stocks_data: pl.DataFrame) -> pl.DataFrame:

    if "asset_returns" in stocks_data.columns:
        return stocks_data

    # Find a suitable price column
    price_cols = [col for col in stocks_data.columns if col in ["prccd", "price", "close", "adj_close"]]

    if not price_cols:
        raise ValueError("No suitable price column found to calculate returns")

    price_col = price_cols[0]

    returns_df = stocks_data.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(1).over("symbol") - 1).alias("asset_returns")
    )

    returns_df = returns_df.filter(pl.col("asset_returns").is_not_null())

    return returns_df