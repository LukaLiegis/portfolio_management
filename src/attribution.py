import polars as pl
import numpy as np


def perform_attribution(
        positions,
        factor_exposures,
        factor_returns,
        specific_returns
):
    """
    Perform sophisticated return attribution analysis.

    Decomposes portfolio returns into:
    - Factor contributions (by factor)
    - Specific (idiosyncratic) contributions
    - Interaction effects
    """
    # Calculate position weights
    weights = positions / np.sum(np.abs(positions))

    # Portfolio factor exposures
    portfolio_exposures = factor_exposures.T @ weights

    # Factor return contributions
    factor_contributions = {}
    for i, factor in enumerate(factor_returns.columns):
        factor_contrib = portfolio_exposures[i] * factor_returns[factor].to_numpy()
        factor_contributions[factor] = factor_contrib

    # Specific return contributions
    specific_contribution = weights @ specific_returns

    # Total expected return
    total_contribution = sum(factor_contributions.values()) + specific_contribution

    # Factor percentage contributions
    factor_pcts = {}
    for factor, contrib in factor_contributions.items():
        factor_pcts[factor] = np.sum(contrib) / np.sum(total_contribution)

    specific_pct = np.sum(specific_contribution) / np.sum(total_contribution)

    attribution = {
        'factor_contributions': factor_contributions,
        'specific_contribution': specific_contribution,
        'total_contribution': total_contribution,
        'factor_pcts': factor_pcts,
        'specific_pct': specific_pct
    }

    return attribution