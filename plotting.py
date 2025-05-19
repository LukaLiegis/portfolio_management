import numpy as np
import matplotlib.pyplot as plt

def visualize_portfolio(portfolio, portfolio_stats):

    if portfolio is None or len(portfolio) == 0:
        print("No portfolio to visualize")
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    portfolio['position'].sort_values().plot(kind='barh', color='skyblue')
    plt.title('Position Sizes')
    plt.xlabel('Position Size ($)')
    plt.ylabel('Stock')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    risk_contrib = (portfolio['position'] * portfolio['idio_vol']) ** 2
    risk_contrib = risk_contrib / risk_contrib.sum()
    risk_contrib.sort_values().plot(kind='barh', color='salmon')
    plt.title('Idiosyncratic Risk Contribution')
    plt.xlabel('Risk Contribution (%)')
    plt.ylabel('Stock')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.scatter(portfolio['alpha'], portfolio['position'],
                alpha=0.7, s=80, c=portfolio['idio_vol'], cmap='viridis')

    for idx, row in portfolio.iterrows():
        plt.annotate(idx, (row['alpha'], row['position']),
                     xytext=(5, 5), textcoords='offset points')

    plt.title('Alpha vs Position Size')
    plt.xlabel('Alpha')
    plt.ylabel('Position Size ($)')
    plt.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
    plt.colorbar(label='Idiosyncratic Volatility')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.bar(['Market'], [portfolio_stats['market_exposure_pct']], color='purple')
    plt.title('Factor Exposures')
    plt.xlabel('Factor')
    plt.ylabel('Exposure (% of GMV)')
    plt.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_performance(backtest_stats, backtest_returns, attribution=None):

    if backtest_returns is None:
        print("No backtest results to visualize")
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    backtest_returns['cumulative_portfolio'].plot(label='Portfolio')
    backtest_returns['cumulative_benchmark'].plot(label='Benchmark')
    backtest_returns['cumulative_market'].plot(label='Market')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if attribution is not None:
        plt.subplot(2, 2, 2)
        attribution['cumulative_factor'].plot(label='Factor')
        attribution['cumulative_idio'].plot(label='Idiosyncratic')
        backtest_returns['cumulative_portfolio'].plot(label='Total')
        plt.title('Return Attribution')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        labels = ['Factor', 'Idiosyncratic']
        sizes = [np.abs(attribution['factor_total']), np.abs(attribution['idio_total'])]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
        plt.axis('equal')
        plt.title('Attribution Breakdown')

    plt.subplot(2, 2, 4)
    stats_labels = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Info Ratio']
    portfolio_stats = [
        backtest_stats['annualized_return'],
        backtest_stats['annualized_vol'],
        backtest_stats['sharpe_ratio'],
        backtest_stats['information_ratio']
    ]
    benchmark_stats = [
        backtest_stats['benchmark_return'],
        backtest_stats['benchmark_vol'],
        backtest_stats['benchmark_sharpe'],
        0
    ]

    x = np.arange(len(stats_labels))
    width = 0.35

    plt.bar(x - width / 2, portfolio_stats, width, label='Portfolio')
    plt.bar(x + width / 2, benchmark_stats, width, label='Benchmark')

    plt.title('Performance Statistics')
    plt.xticks(x, stats_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()