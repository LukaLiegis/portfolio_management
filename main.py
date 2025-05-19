from data import get_data
from alpha import estimate_alphas
from config import STOCKS, BENCHMARKS
from backtest import backtest_portfolio
from portfolio import construct_portfolio
from attribution import perform_attribution
from factor_model import create_factor_model
from plotting import visualize_portfolio, visualize_performance


def run_analysis(stocks, start_date, end_date, benchmark, target_gmv, sizing_method,
                 max_factor_exposure, max_stock_weight):

    price_data, returns_data, market_cap, momentum = get_data(start_date, end_date, stocks,  benchmark)

    factor_betas, idio_vol, factor_r2 = create_factor_model(stocks, returns_data)

    alphas = estimate_alphas(stocks, returns_data)

    portfolio, portfolio_stats = construct_portfolio(
        alphas, factor_betas, idio_vol, target_gmv, sizing_method, max_factor_exposure, max_stock_weight
    )

    backtest_stats, backtest_returns = backtest_portfolio(portfolio, returns_data, benchmark)

    attribution = perform_attribution(portfolio_stats, backtest_returns)

    visualize_portfolio(portfolio, portfolio_stats)
    visualize_performance(backtest_stats, backtest_returns, attribution)

    print("\nPortfolio Summary:")
    print("-----------------")
    print(f"GMV: ${portfolio_stats['gmv']:,.2f}")
    print(f"Market Exposure: {portfolio_stats['market_exposure_pct']:.2%}")
    print(f"Idiosyncratic Volatility: {portfolio_stats['idio_volatility']:,.2f}")
    print(f"Total Volatility: {portfolio_stats['total_volatility']:,.2f}")
    print(f"% Idiosyncratic Variance: {portfolio_stats['pct_idio_var']:.2%}")
    print(f"Expected Return (based on alpha): {portfolio_stats['expected_return']:,.2f}")

    print("\nBacktest Performance:")
    print("---------------------")
    print(f"Annualized Return: {backtest_stats['annualized_return']:.2%}")
    print(f"Annualized Volatility: {backtest_stats['annualized_vol']:.2%}")
    print(f"Sharpe Ratio: {backtest_stats['sharpe_ratio']:.2f}")
    print(f"Information Ratio: {backtest_stats['information_ratio']:.2f}")

    return portfolio, portfolio_stats, backtest_stats

if __name__ == "__main__":
    portfolio, portfolio_stats, backtest_stats = run_analysis(
        stocks = STOCKS,
        benchmark = BENCHMARKS,
        start_date = '2010-01-01',
        end_date = '2024-12-31',
        target_gmv=1000000,  # $1M portfolio
        sizing_method='proportional',
        max_factor_exposure=0.2,  # 20% max market exposure
        max_stock_weight=0.15,  # 15% max position size
    )