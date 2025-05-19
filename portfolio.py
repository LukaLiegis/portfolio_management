from sizing import size_proportions


def construct_portfolio(alphas, factor_betas, idio_vol, target_gmv,
                        sizing_method, max_factor_exposure,
                        max_stock_weight):
    print("Constructing portfolio...")

    # Size positions based on alphas
    positions = size_proportions()