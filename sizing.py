import numpy as np
import pandas as pd

def size_positions(alphas, idio_vol, factor_betas, target_gmv,
                     sizing_method = 'proportional', risk_aversion = 1.0):

    common_tickers = list(set(factor_betas.index) &
                          set(idio_vol.index) &
                          set(alphas.index))

    filtered_alphas = alphas[common_tickers]
    filtered_idio_vol = idio_vol[common_tickers]

    if sizing_method == 'proportional':
        raw_positions = filtered_alphas

    elif sizing_method == 'risk_parity':
        raw_positions = filtered_alphas / filtered_idio_vol

    elif sizing_method == 'mean_variance':
        raw_positions = filtered_alphas / (filtered_idio_vol ** 2 * risk_aversion)

    else:
        print(f'Unknown sizing method: {sizing_method}')
        return None

    raw_positions = raw_positions.fillna(0)

    if np.sum(np.abs(raw_positions)) > 0:
        scale_factor = target_gmv / np.sum(np.abs(raw_positions))
        positions = raw_positions * scale_factor
    else:
        positions = pd.Series(0, index=common_tickers)

    return positions