STOCKS = [
    'VRTX',
    'SMMT',
    'RPRX',
    'REGN',
    'ONC',
    'INSM',
    'INCY',
    'GMAB',
    'EXEL',
    'BNTX',
    'BMRN',
    'ARGX',
    'ALNY',
]

BENCHMARKS = [
    'SPY',
    'IBB',
]

DEFAULT_SETTINGS = {
    'target_gmv': 1000000,        # $1M portfolio
    'max_stock_weight': 0.15,     # 15% max position size
    'risk_aversion': 1.0,         # Risk aversion parameter
    'lookback_days': 252,         # Lookback for historical analysis
    'rebalance_frequency': 21,
}

FACTOR_CONSTRAINTS = {
    # Toraniko factors
    'market': 0.2,  # ±20% net market exposure
    'Technology': 0.1,  # ±10% sector exposure
    'Healthcare': 0.1,
    'Financial': 0.1,
    'Consumer': 0.1,
    'Industrial': 0.1,
    'Energy': 0.1,
    'Utilities': 0.1,
    'Materials': 0.1,
    'Communication': 0.1,
    'RealEstate': 0.1,
    'momentum': 0.3,  # ±30% style factor exposure
    'size': 0.3,
    'value': 0.3,

    # Fama-French factors
    'mktrf': 0.2,  # Market (±20%)
    'smb': 0.3,  # Size (±30%)
    'hml': 0.3,  # Value (±30%)
    'rmw': 0.3,  # Profitability (±30%)
    'cma': 0.3,  # Investment (±30%)
    'umd': 0.3  # Momentum (±30%)
}