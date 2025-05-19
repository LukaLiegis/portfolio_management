import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import matplotlib.dates as mdates


def plot_factor_exposures(portfolio_stats_history):
    """
    Plot factor exposures over time.
    """
    dates = [stats['date'] for stats in portfolio_stats_history]
    factors = list(portfolio_stats_history[0]['factor_exposures'].keys())

    # Prepare data
    factor_exposures = {factor: [] for factor in factors}
    for stats in portfolio_stats_history:
        for factor in factors:
            factor_exposures[factor].append(stats['factor_exposures'][factor])

    # Create plot
    plt.figure(figsize=(12, 8))
    for factor, exposures in factor_exposures.items():
        plt.plot(dates, exposures, label=factor, linewidth=2)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Factor Exposures Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Exposure', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return plt.gcf()


def plot_risk_decomposition(portfolio_stats_history):
    """
    Plot risk decomposition over time.
    """
    dates = [stats['date'] for stats in portfolio_stats_history]
    factor_risk = [stats['factor_risk'] for stats in portfolio_stats_history]
    specific_risk = [stats['specific_risk'] for stats in portfolio_stats_history]
    total_risk = [stats['total_risk'] for stats in portfolio_stats_history]

    plt.figure(figsize=(12, 8))

    plt.stackplot(dates,
                  [np.array(factor_risk) ** 2, np.array(specific_risk) ** 2],
                  labels=['Factor Risk', 'Specific Risk'],
                  colors=['#4CAF50', '#F44336'],
                  alpha=0.7)

    plt.plot(dates, total_risk, 'k--', label='Total Risk', linewidth=2)

    plt.title('Portfolio Risk Decomposition', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Risk (Annualized Volatility)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return plt.gcf()


def create_performance_report(backtest_results, attribution_results, benchmark_returns=None):
    """
    Generate comprehensive performance report.
    """
    # Create performance dashboard
    fig = plt.figure(figsize=(15, 12))

    # 1. Cumulative return plot
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(backtest_results['dates'], backtest_results['cumulative_returns'],
             label='Strategy', linewidth=2)
    if benchmark_returns is not None:
        ax1.plot(backtest_results['dates'], benchmark_returns,
                 label='Benchmark', linewidth=2, linestyle='--')
    ax1.set_title('Cumulative Performance', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Factor attribution
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    factors = list(attribution_results['factor_pcts'].keys())
    factor_pcts = [attribution_results['factor_pcts'][f] for f in factors]
    factor_pcts.append(attribution_results['specific_pct'])
    factors.append('Specific')

    colors = plt.cm.viridis(np.linspace(0, 1, len(factors)))
    ax2.pie(factor_pcts, labels=factors, autopct='%1.1f%%',
            startangle=90, colors=colors)
    ax2.set_title('Return Attribution', fontsize=14)

    # 3. Rolling Sharpe ratio
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.plot(backtest_results['dates'][63:], backtest_results['rolling_sharpe'],
             label='Rolling Sharpe (3m)', linewidth=2)
    ax3.axhline(y=backtest_results['sharpe_ratio'], color='r',
                linestyle='--', label=f'Overall Sharpe: {backtest_results["sharpe_ratio"]:.2f}')
    ax3.set_title('Rolling Sharpe Ratio', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Drawdown chart
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.fill_between(backtest_results['dates'], backtest_results['drawdowns'],
                     0, color='red', alpha=0.3)
    ax4.set_title('Drawdowns', fontsize=14)
    ax4.set_ylim(min(backtest_results['drawdowns']) * 1.1, 0)
    ax4.grid(alpha=0.3)

    # 5. Monthly returns heatmap
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    # [Code for monthly returns heatmap]

    plt.tight_layout()
    return fig