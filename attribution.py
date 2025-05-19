def perform_attribution(portfolio_stats, backtest_returns):
    """
    Perform performance attribution analysis
    Decompose returns into factor and idiosyncratic components

    Parameters:
    -----------
    portfolio : DataFrame
        Portfolio with positions and weights
    portfolio_stats : dict
        Portfolio statistics
    backtest_returns : dict
        Backtest return data

    Returns:
    --------
    dict: Attribution results
    """
    print("Performing attribution analysis...")

    if backtest_returns is None:
        print("No backtest results for attribution")
        return None

    portfolio_returns = backtest_returns['portfolio']
    market_returns = backtest_returns['market']

    market_exposure = portfolio_stats['market_exposure_pct']
    factor_contribution = market_exposure * market_returns

    idio_contribution = portfolio_returns - factor_contribution

    cumulative_factor = (1 + factor_contribution).cumprod() - 1
    cumulative_idio = (1 + idio_contribution).cumprod() - 1

    factor_total = factor_contribution.sum()
    idio_total = idio_contribution.sum()
    total_return = portfolio_returns.sum()

    factor_pct = factor_total / total_return if total_return != 0 else 0
    idio_pct = idio_total / total_return if total_return != 0 else 0

    attribution = {
        'factor_contribution': factor_contribution,
        'idio_contribution': idio_contribution,
        'cumulative_factor': cumulative_factor,
        'cumulative_idio': cumulative_idio,
        'factor_total': factor_total,
        'idio_total': idio_total,
        'factor_pct': factor_pct,
        'idio_pct': idio_pct
    }

    return attribution