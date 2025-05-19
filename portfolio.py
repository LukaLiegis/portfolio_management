import numpy as np
import pandas as pd
from sizing import size_positions
from remove_risk import neutralize_factor_risk


def construct_portfolio(alphas, factor_betas, idio_vol, target_gmv,
                        sizing_method, max_factor_exposure,
                        max_stock_weight):
    print("Constructing portfolio...")

    # Size positions based on alphas
    positions = size_positions(alphas, idio_vol, factor_betas, target_gmv, sizing_method)

    # Apply single-stock concentration limits
    if positions is not None and len(positions) > 0:
        current_gmv = np.sum(np.abs(positions))
        max_positions = current_gmv * max_stock_weight

        for ticker in positions.index:
            if np.abs(positions[ticker]) > max_positions:
                positions[ticker] = np.sign(positions[ticker]) * max_positions

    positions = neutralize_factor_risk(positions, factor_betas, max_factor_exposure)

    portfolio = pd.DataFrame({
        'positions': positions,
        'weight': positions / np.sum(np.abs(positions)),
        'alpha': alphas[positions.index],
        'market_beta': factor_betas[positions.index],
        'idio_vol': idio_vol[positions.index],
    })

    # Calculate portfolio stats
    gmv = np.sum(np.abs(portfolio['positions']))
    market_exposure = np.sum(portfolio['positions'] * portfolio['market_beta'])

    # Idio risk
    idio_variance = np.sum((portfolio['positions'] * portfolio['idio_vol']) ** 2)
    idio_volatility = np.sqrt(idio_variance)

    # TODO: Calculate actual market volatility rather than taking an assumption.
    market_vol = 0.15
    factor_volatility = np.abs(market_exposure) * market_vol

    total_volatility = np.sqrt(idio_variance + market_exposure ** 2 * market_vol ** 2)

    pct_idio_var = idio_variance / (idio_variance + market_exposure ** 2 * market_vol ** 2)

    expected_return = np.sum(portfolio['positions'] * portfolio['alpha'])

    portfolio_stats = pd.DataFrame({
        'gmv': gmv,
        'market_exposure': market_exposure,
        'market_exposure_pct': market_exposure / gmv,
        'idio_volatility': idio_volatility,
        'factor_volatility': factor_volatility,
        'total_volatility': total_volatility,
        'pct_idio_var': pct_idio_var,
        'expected_return': expected_return,
    })

    return portfolio, portfolio_stats
