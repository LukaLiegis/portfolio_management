import polars as pl


def load_and_process_data(stock_files, factor_file, start_date, end_date):
    """
    Load and process stock and factor data using polars for performance.
    """
    stocks_df = pl.read_csv(stock_files)

    stocks_data = stocks_df.select([
        "tic", "datadate", "prccd", "prchd", "prcld", "cshoc",
        "eps", "gsector", "gind", "gsubind", "sic",
        "cshtrd", "div", "ajexdi", "exchg", "trfd"
    ])

    stocks_data = stocks_data.with_columns([
        # Market cap (shares outstanding * price)
        (pl.col("cshoc") * pl.col("prccd")).alias("market_cap"),

        # Book-to-price (inverse of P/E ratio)
        (pl.col("eps") / pl.col("prccd")).alias("book_price"),

        # Turnover (trading volume / shares outstanding)
        (pl.col("cshtrd") / pl.col("cshoc")).alias("turnover"),

        # Dividend yield
        (pl.col("div") / pl.col("prccd")).alias("div_yield")
    ])

    factors_df = pl.read_csv(factor_file)

    return stocks_data, factors_df