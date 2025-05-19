import polars as pl
from pathlib import Path

from src.data import load_and_process_data
from src.factors import factor_mom, factor_size, factor_value, factor_quality
from src.risk_model import build_risk_model
from src.portfolio import construct_portfolio, calculate_returns
from src.backtest import backtest_strategy
from src.attribution import perform_attribution
from src.plotting import (
    plot_factor_exposures,
    plot_risk_decomposition,
    create_performance_report
)


def run_portfolio_system(
        stock_files,
        factor_file,
        start_date,
        end_date,
        initial_capital=10_000_000,
        rebalance_frequency=21,
        max_position=0.15,
        output_dir="./output"
):
    """
    Run the complete biotech portfolio management system.
    """
    print("Loading and processing data...")
    stocks_data, factors_data = load_and_process_data(
        stock_files, factor_file, start_date, end_date
    )

    # Create directory for outputs
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("Calculating factor scores...")
    # Calculate factor scores
    returns_data = calculate_returns(stocks_data)

    mom_scores = factor_mom(returns_data)
    size_scores = factor_size(stocks_data)
    value_scores = factor_value(stocks_data)
    quality_scores = factor_quality(stocks_data, returns_data)

    # Combine factor scores
    factor_scores = (
        mom_scores
        .join(size_scores, on=["date", "symbol"])
        .join(value_scores, on=["date", "symbol"])
        .join(quality_scores, on=["date", "symbol"])
    )

    print("Building risk model...")
    # Build risk model
    exposures, factor_cov, specific_risk = build_risk_model(
        returns_data, factors_data, stocks_data.select("market_cap")
    )

    # Define strategy function for backtesting
    def biotech_strategy(current_stocks, current_factors, current_date):
        """Strategy function that generates positions for a given date."""
        # Get latest factor scores
        latest_scores = factor_scores.filter(pl.col("date") <= current_date).sort("date").group_by("symbol").last()

        # Create alpha signal from factor scores
        alphas = (
                latest_scores["mom_score"] * 0.3 +
                latest_scores["size_score"] * 0.2 +
                latest_scores["value_score"] * 0.3 +
                latest_scores["quality_score"] * 0.2
        )

        # Get latest risk model data
        # Simplified for explanation - would need proper time filtering
        latest_exposures = exposures
        latest_factor_cov = factor_cov
        latest_specific_risk = specific_risk

        # Construct portfolio
        positions, stats = construct_portfolio(
            alphas,
            latest_exposures,
            latest_factor_cov,
            latest_specific_risk,
            initial_capital,
            max_position
        )

        return positions, stats

    print("Running backtest...")
    backtest_results = backtest_strategy(
        initial_capital,
        stocks_data,
        factors_data,
        biotech_strategy,
        rebalance_frequency,
        start_date=start_date,
        end_date=end_date
    )

    print("Performing attribution analysis...")
    attribution_results = perform_attribution(
        backtest_results[-1]["positions"],
        exposures,
        factors_data,
        returns_data.filter(pl.col("date") > backtest_results[-1]["date"]),
        stocks_data.select("symbol").unique()
    )

    print("Generating reports...")
    factor_plot = plot_factor_exposures([r["portfolio_stats"] for r in backtest_results if "portfolio_stats" in r])
    factor_plot.savefig(output_path / "factor_exposures.png")

    risk_plot = plot_risk_decomposition([r["portfolio_stats"] for r in backtest_results if "portfolio_stats" in r])
    risk_plot.savefig(output_path / "risk_decomposition.png")

    perf_report = create_performance_report(backtest_results, attribution_results)
    perf_report.savefig(output_path / "performance_report.png")

    print(f"Analysis complete. Results saved to {output_path}")

    return backtest_results, attribution_results


if __name__ == "__main__":
    run_portfolio_system(
        stock_files=["vrtx_example.csv"],
        factor_file="data/w1fs4jrazjijad60.csv.gz",
        start_date="2000-01-01",
        end_date="2024-12-31",
        initial_capital=10_000_000,
        rebalance_frequency=21,
        max_position=0.15,
        output_dir="./portfolio_results"
    )