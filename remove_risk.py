import numpy as np

def neutralize_factor_risk(positions, factor_betas, max_factor_exposure: float = 2.0):
    if positions is None or len(positions) == 0:
        print('No positions to neutralize...')
        return None

    common_tickers = list(set(positions.index) & set(factor_betas.index))
    positions = positions[common_tickers]
    betas = factor_betas[common_tickers]

    market_exposure = np.sum(positions * betas)

    current_gmv = np.sum(np.abs(positions))

    max_exposure = current_gmv * max_factor_exposure

    if np.abs(market_exposure) > max_exposure:
        print(f"Market exposure ({market_exposure:.2f}) exceeds limit ({max_exposure:.2f})")

        hedge_needed = market_exposure - np.sign(market_exposure) * max_exposure

        beta_contribution = positions * betas
        beta_contribution_sum = np.sum(np.abs(beta_contribution))

        adjustment = hedge_needed / beta_contribution_sum

        adjusted_positions = positions - beta_contribution * adjustment

        new_market_exposure = np.sum(adjusted_positions * betas)
        print(f"New market exposure: {new_market_exposure:.2f}")

        return adjusted_positions

    return positions